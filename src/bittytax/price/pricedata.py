# -*- coding: utf-8 -*-
# (c) Nano Nano Ltd 2019

import os
from decimal import Decimal
from typing import List, Optional, Tuple

from colorama import Fore

from ..bt_types import (
    AssetName,
    AssetSymbol,
    DataSourceName,
    Date,
    QuoteSymbol,
    SourceUrl,
    Timestamp,
    TradingPair,
)
from ..config import config
from ..constants import CACHE_DIR
from .datasource import DataSourceBase
from .exceptions import UnexpectedDataSourceError


class PriceData:
    def __init__(
        self, data_sources_required: List[DataSourceName], price_tool: bool = False
    ) -> None:
        self.price_tool = price_tool
        self.data_sources = {}

        if not os.path.exists(CACHE_DIR):
            os.mkdir(CACHE_DIR)

        for data_source_class in DataSourceBase.__subclasses__():
            if data_source_class.__name__.upper() in [ds.upper() for ds in data_sources_required]:
                self.data_sources[data_source_class.__name__.upper()] = data_source_class()

    @staticmethod
    def data_source_priority(asset: AssetSymbol) -> List[DataSourceName]:
        if asset in config.data_source_select:
            return [ds.split(":")[0] for ds in config.data_source_select[asset]]
        if asset in config.fiat_list:
            return config.data_source_fiat
        return config.data_source_crypto

    def get_latest_ds(
        self, data_source: DataSourceName, asset: AssetSymbol, quote: QuoteSymbol
    ) -> Tuple[Optional[Decimal], AssetName]:
        if data_source.upper() in self.data_sources:
            if asset in self.data_sources[data_source.upper()].assets:
                return (
                    self.data_sources[data_source.upper()].get_latest(asset, quote),
                    self.data_sources[data_source.upper()].assets[asset]["name"],
                )

            return None, AssetName("")
        raise UnexpectedDataSourceError(data_source, DataSourceBase.datasources_str())

    def get_historical_ds(
        self,
        data_source: DataSourceName,
        asset: AssetSymbol,
        quote: QuoteSymbol,
        timestamp: Timestamp,
        no_cache: bool = False,
    ) -> Tuple[Optional[Decimal], AssetName, SourceUrl]:
        # Debug: Inizio funzione
        if config.debug:
            print(f"DEBUG: Inizio get_historical_ds per asset: {asset}, quote: {quote}, timestamp: {timestamp}, no_cache: {no_cache}, data_source: {data_source}")

        # Verifica se il data source è tra quelli disponibili
        if data_source.upper() in self.data_sources:
            if config.debug:
                print(f"DEBUG: Data source {data_source} trovato")

            # Rimuovi eventuali spazi bianchi dal simbolo dell'asset
            asset = asset.strip().upper()

            # Verifica se l'asset è supportato dal data source
            if asset in self.data_sources[data_source.upper()].assets:
                if config.debug:
                    print(f"DEBUG: Asset {asset} trovato nel data source {data_source}")

                date = Date(timestamp.date())
                pair = TradingPair(asset + "/" + quote)

                # Controlla la cache se no_cache è False
                if not no_cache:
                    if config.debug:
                        print(f"DEBUG: Controllo cache per la coppia {pair} e la data {date}")

                    # Controllo se la cache ha il prezzo per la coppia e la data specificati
                    if (
                        pair in self.data_sources[data_source.upper()].prices
                        and date in self.data_sources[data_source.upper()].prices[pair]
                    ):
                        if config.debug:
                            print(f"DEBUG: Prezzo trovato in cache per {pair} a {date}")
                    
                        return (
                            self.data_sources[data_source.upper()].prices[pair][date]["price"],
                            self.data_sources[data_source.upper()].assets[asset]["name"],
                            self.data_sources[data_source.upper()].prices[pair][date]["url"],
                        )

                # Se non è in cache, richiede i dati storici
                if config.debug:
                    print(f"DEBUG: Chiamata per ottenere il prezzo storico da {data_source} per {asset} contro {quote} a {timestamp}")

                self.data_sources[data_source.upper()].get_historical(asset, quote, timestamp)

                # Dopo la richiesta, verifica nuovamente la presenza dei dati
                if (
                    pair in self.data_sources[data_source.upper()].prices
                    and date in self.data_sources[data_source.upper()].prices[pair]
                ):
                    if config.debug:
                        print(f"DEBUG: Prezzo ottenuto con successo da {data_source} per {pair} a {date}")

                    return (
                        self.data_sources[data_source.upper()].prices[pair][date]["price"],
                        self.data_sources[data_source.upper()].assets[asset]["name"],
                        self.data_sources[data_source.upper()].prices[pair][date]["url"],
                    )

                # Se il prezzo non viene trovato, restituisce None
                if config.debug:
                    print(f"DEBUG: Nessun prezzo trovato per {pair} a {date} dopo la chiamata storica")

                return (
                    None,
                    self.data_sources[data_source.upper()].assets[asset]["name"],
                    SourceUrl(""),
                )

            # Se l'asset non è supportato dal data source
            if config.debug:
                print(f"DEBUG: Asset {asset} non trovato nel data source {data_source}")

            return None, AssetName(""), SourceUrl("")
    
        # Data source non valido
        if config.debug:
            print(f"DEBUG: Data source {data_source} non trovato tra quelli disponibili")

        raise UnexpectedDataSourceError(data_source, DataSourceBase.datasources_str())


    def get_latest(
        self, asset: AssetSymbol, quote: QuoteSymbol
    ) -> Tuple[Optional[Decimal], AssetName, DataSourceName]:
        name = AssetName("")
        for data_source in self.data_source_priority(asset):
            price, name = self.get_latest_ds(data_source, asset, quote)
            if price is not None:
                if config.debug:
                    print(
                        f"{Fore.YELLOW}price: <latest>, 1 "
                        f"{asset}={price.normalize():0,f} {quote} via "
                        f"{self.data_sources[data_source.upper()].name()} ({name})"
                    )
                if self.price_tool:
                    print(
                        f"{Fore.YELLOW}1 {asset}={price.normalize():0,f} {quote} "
                        f"{Fore.CYAN}via {self.data_sources[data_source.upper()].name()} ({name})"
                    )
                return price, name, self.data_sources[data_source.upper()].name()
        return None, name, DataSourceName("")

    def get_historical(
        self, asset: AssetSymbol, quote: QuoteSymbol, timestamp: Timestamp, no_cache: bool = False
    ) -> Tuple[Optional[Decimal], AssetName, DataSourceName, SourceUrl]:
        name = AssetName("")

        # Debug: Inizio funzione
        if config.debug:
            print(f"DEBUG: Inizio get_historical per asset: {asset}, quote: {quote}, timestamp: {timestamp}, no_cache: {no_cache}")

        # Ciclo attraverso le fonti dati in ordine di priorità
        for data_source in self.data_source_priority(asset):
            if config.debug:
                print(f"DEBUG: Prova a ottenere dati storici da {data_source} per {asset} contro {quote}")

            # Richiesta del prezzo storico per il data_source corrente
            price, name, url = self.get_historical_ds(data_source, asset, quote, timestamp, no_cache)

            # Debug: Controllo se il prezzo è stato trovato
            if config.debug:
                if price is not None:
                    print(f"DEBUG: Prezzo trovato: {asset}={price.normalize():0,f} {quote} da {data_source}")
                else:
                    print(f"DEBUG: Nessun prezzo trovato per {asset} contro {quote} da {data_source} a {timestamp}")

            # Se il prezzo è stato trovato, restituiscilo immediatamente
            if price is not None:
                if config.debug:
                    print(
                        f"{Fore.YELLOW}price: {timestamp:%Y-%m-%d}, 1 {asset}={price.normalize():0,f} {quote} via "
                        f"{self.data_sources[data_source.upper()].name()} ({name})"
                    )
                if self.price_tool:
                    print(
                        f"{Fore.YELLOW}1 {asset}={price.normalize():0,f} {quote} "
                        f"{Fore.CYAN}via {self.data_sources[data_source.upper()].name()} ({name})"
                    )
                return price, name, self.data_sources[data_source.upper()].name(), url

        # Debug: Nessun prezzo trovato dopo aver controllato tutte le fonti dati
        if config.debug:
            print(f"DEBUG: Nessun prezzo trovato per {asset} contro {quote} a {timestamp} da nessuna fonte dati.")

        return None, name, DataSourceName(""), SourceUrl("")
