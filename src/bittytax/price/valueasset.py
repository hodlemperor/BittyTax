# -*- coding: utf-8 -*-
# (c) Nano Nano Ltd 2019

from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional, Tuple

from colorama import Fore, Style
from tqdm import tqdm
from typing_extensions import TypedDict

from ..bt_types import (
    AssetName,
    AssetSymbol,
    DataSourceName,
    Date,
    FixedValue,
    QuoteSymbol,
    SourceUrl,
    Timestamp,
    Year,
)
from ..config import config
from ..constants import WARNING
from .pricedata import PriceData
import time

class VaPriceReport(TypedDict):  # pylint: disable=too-few-public-methods
    name: AssetName
    data_source: DataSourceName
    url: SourceUrl
    price_ccy: Optional[Decimal]
    price_btc: Optional[Decimal]


class ValueAsset:
    def __init__(self, price_tool: bool = False) -> None:
        self.price_tool = price_tool
        self.price_report: Dict[Year, Dict[AssetSymbol, Dict[Date, VaPriceReport]]] = {}
        data_sources_required = set(config.data_source_fiat + config.data_source_crypto) | {
            x.split(":")[0] for v in config.data_source_select.values() for x in v
        }
        self.price_data = PriceData(list(data_sources_required), price_tool)

    def get_value(
        self, asset: AssetSymbol, timestamp: Timestamp, quantity: Decimal
    ) -> Tuple[Decimal, FixedValue]:
        if asset == config.ccy:
            return quantity, FixedValue(True)

        if quantity == 0:
            return Decimal(0), FixedValue(False)

        asset_price_ccy, _, _ = self.get_historical_price(asset, timestamp)
        if asset_price_ccy is not None:
            value = asset_price_ccy * quantity
            if config.debug:
                print(
                    f"{Fore.YELLOW}price: {timestamp:%Y-%m-%d}, 1 "
                    f"{asset}={config.sym()}{asset_price_ccy:0,.2f} {config.ccy}, "
                    f"{quantity.normalize():0,f} {asset}="
                    f"{Style.BRIGHT}{config.sym()}{value:0,.2f} {config.ccy}{Style.NORMAL}"
                )
            return value, FixedValue(False)

        tqdm.write(
            f"{WARNING} Price for {asset} on {timestamp:%Y-%m-%d} is not available, "
            f"using price of {config.sym()}{0:0,.2f}"
        )
        return Decimal(0), FixedValue(False)

    def get_current_value(
        self, asset: AssetSymbol, quantity: Decimal
    ) -> Tuple[Optional[Decimal], AssetName, DataSourceName]:
        asset_price_ccy, name, data_source = self.get_latest_price(asset)
        if asset_price_ccy is not None:
            return asset_price_ccy * quantity, name, data_source

        return None, AssetName(""), DataSourceName("")

    def get_historical_price(
        self, asset: AssetSymbol, timestamp: Timestamp, no_cache: bool = False
    ) -> Tuple[Optional[Decimal], AssetName, DataSourceName]:
        asset_price_ccy = None

        if not self.price_tool and timestamp.date() >= datetime.now().date():
            tqdm.write(
                f"{WARNING} Price for {asset} on {timestamp:%Y-%m-%d}, "
                f"no historic price available, using latest price"
            )
            return self.get_latest_price(asset)

        if asset == "BTC" or asset in config.fiat_list:
            asset_price_ccy, name, data_source, url = self.price_data.get_historical(
                asset, config.ccy, timestamp, no_cache
            )
            self.price_report_cache(asset, timestamp, name, data_source, url, asset_price_ccy)
        else:
            asset_price_btc, name, data_source, url = self.price_data.get_historical(
                asset, QuoteSymbol("BTC"), timestamp, no_cache
            )
            if asset_price_btc is not None:
                (
                    btc_price_ccy,
                    name2,
                    data_source2,
                    url2,
                ) = self.price_data.get_historical(
                    AssetSymbol("BTC"), config.ccy, timestamp, no_cache
                )
                if btc_price_ccy is not None:
                    asset_price_ccy = btc_price_ccy * asset_price_btc

                self.price_report_cache(
                    AssetSymbol("BTC"), timestamp, name2, data_source2, url2, btc_price_ccy
                )

            self.price_report_cache(
                asset,
                timestamp,
                name,
                data_source,
                url,
                asset_price_ccy,
                asset_price_btc,
            )

        return asset_price_ccy, name, data_source

    def get_latest_price(
        self, asset: AssetSymbol
    ) -> Tuple[Optional[Decimal], AssetName, DataSourceName]:
        asset_price_ccy = None

        if asset == "BTC" or asset in config.fiat_list:
            asset_price_ccy, name, data_source = self.price_data.get_latest(asset, config.ccy)
        else:
            asset_price_btc, name, data_source = self.price_data.get_latest(
                asset, QuoteSymbol("BTC")
            )

            if asset_price_btc is not None:
                btc_price_ccy, _, _ = self.price_data.get_latest(AssetSymbol("BTC"), config.ccy)
                if btc_price_ccy is not None:
                    asset_price_ccy = btc_price_ccy * asset_price_btc

        return asset_price_ccy, name, data_source

    def price_report_cache(
        self,
        asset: AssetSymbol,
        timestamp: Timestamp,
        name: AssetName,
        data_source: DataSourceName,
        url: SourceUrl,
        price_ccy: Optional[Decimal],
        price_btc: Optional[Decimal] = None,
    ) -> None:
        date = timestamp.date()

        if date > config.get_tax_year_end(date.year):
            tax_year = Year(date.year + 1)
        else:
            tax_year = Year(date.year)

        if tax_year not in self.price_report:
            self.price_report[tax_year] = {}

        if asset not in self.price_report[tax_year]:
            self.price_report[tax_year][asset] = {}

        if date not in self.price_report[tax_year][asset]:
            self.price_report[tax_year][asset][Date(date)] = VaPriceReport(
                name=name,
                data_source=data_source,
                url=url,
                price_ccy=price_ccy,
                price_btc=price_btc,
            )

    # Funzione per gestire la richiesta con retry, verifica delle chiavi e fallback
    def get_data_with_retry(self, asset, quote, timestamp, attempts=3, delay=5, no_cache=False):
        for attempt in range(attempts):
            try:
                # Effettua la richiesta per ottenere i dati
                json_resp = self.price_data.get_historical(asset, quote, timestamp, no_cache)

                # Verifica che le chiavi "Response" e "Type" siano presenti
                if "Response" in json_resp and json_resp["Response"] == "Success" and json_resp.get("Type") == 2:
                    # Se la risposta è valida, restituisci i dati richiesti
                    return json_resp["Data"], json_resp.get("Name"), json_resp.get("Url")
                else:
                    # Se la risposta non è valida, stampa un messaggio di errore
                    print(f"Formato di risposta inatteso: {json_resp}")
                    raise KeyError("Risposta non valida")
        
            except KeyError as e:
                # Log dell'errore e tentativo di retry
                print(f"Errore durante il tentativo {attempt + 1}: {e}. Ritento tra {delay} secondi...")
                time.sleep(delay)
    
        # Se tutti i tentativi falliscono, utilizza una logica di fallback
        print(f"Impossibile ottenere dati per {asset} in {quote} alla data {timestamp}. Utilizzo dei valori di fallback.")
        return None, None, None

    def get_historical_price_with_conversion(
        self, asset: AssetSymbol, timestamp: Timestamp, target_ccy: AssetSymbol, no_cache: bool = False
    ) -> Tuple[Optional[Decimal], AssetName, DataSourceName]:
        """
        Ottiene il prezzo storico di un asset rispetto a una valuta target (come EUR).
    
        :param asset: Simbolo dell'asset (ad esempio BTC).
        :param timestamp: Il timestamp per ottenere il prezzo storico.
        :param target_ccy: Valuta di riferimento (ad esempio EUR).
        :param no_cache: Booleano per disabilitare la cache.
        :return: Il prezzo storico nella valuta richiesta, il nome dell'asset e la fonte dei dati.
        """
        # Prova a ottenere il prezzo storico dell'asset rispetto a BTC o alla valuta configurata
        asset_price_btc_or_ccy, name, url = self.get_data_with_retry(asset, QuoteSymbol("BTC") if asset != "BTC" else target_ccy, timestamp, no_cache=no_cache)

        # Se il prezzo è disponibile direttamente in target_ccy (esempio BTC/EUR)
        if asset == "BTC" or asset in config.fiat_list:
            return asset_price_btc_or_ccy, name, url

        # Se è necessaria la conversione da BTC alla valuta target
        if asset_price_btc_or_ccy is not None:
            # Ottieni il prezzo di BTC nella valuta target (ad esempio BTC/EUR)
            btc_to_target_price, name2, url2 = self.get_data_with_retry(AssetSymbol("BTC"), target_ccy, timestamp, no_cache=no_cache)
            if btc_to_target_price is not None:
                # Converti il prezzo dell'asset nella valuta target
                asset_price_in_target_ccy = asset_price_btc_or_ccy * btc_to_target_price
                return asset_price_in_target_ccy, name2, url2

        # Se non è possibile ottenere il prezzo, utilizza valori di fallback
        return None, name, url
