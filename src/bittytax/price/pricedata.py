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
from datetime import date  

import json

CACHE_FILE_PATH = os.path.join(CACHE_DIR, "failed_requests.json")  # Usa CACHE_DIR per la cache

class PriceData:
    def __init__(
        self, data_sources_required: List[DataSourceName], price_tool: bool = False
    ) -> None:
        self.price_tool = price_tool
        self.data_sources = {}
        self.failed_requests = self.load_failed_requests()

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

    # Funzione per caricare le richieste fallite
    def load_failed_requests(self):
        if os.path.exists(CACHE_FILE_PATH):
            if os.stat(CACHE_FILE_PATH).st_size == 0:
                return set()
            with open(CACHE_FILE_PATH, "r") as f:
                try:
                    return set(tuple(x[:-1]) + (date.fromisoformat(x[-1]),) for x in json.load(f))
                except json.JSONDecodeError:
                    print("Errore di decodifica JSON. Ricreo il file 'failed_requests.json'.")
                    os.remove(CACHE_FILE_PATH)  # Elimina il file corrotto
                    return set()  # Restituisce un set vuoto
        return set()

    # Funzione per salvare le richieste fallite
    def save_failed_requests(self):
        with open(CACHE_FILE_PATH, "w") as f:
            json.dump([list(item[:-1]) + [item[-1].isoformat()] if isinstance(item[-1], date) else list(item) for item in self.failed_requests], f)

    def get_historical_ds(
        self,
        data_source: DataSourceName,
        asset: AssetSymbol,
        quote: QuoteSymbol,
        timestamp: Timestamp,
        no_cache: bool = False,
    ) -> Tuple[Optional[Decimal], AssetName, SourceUrl]:
        date = Date(timestamp.date())
        pair = TradingPair(asset + "/" + quote)

        # Controlla se questa richiesta fallita
        if (data_source, asset, quote, date) in self.failed_requests:
            print(f"Skipping request for {asset} on {date} (previous failure).")
            return None, AssetName(""), SourceUrl("")

        if data_source.upper() in self.data_sources:
            if asset in self.data_sources[data_source.upper()].assets:
                if not no_cache:
                    # Controlla la cache
                    if (
                        pair in self.data_sources[data_source.upper()].prices
                        and date in self.data_sources[data_source.upper()].prices[pair]
                    ):
                        return (
                            self.data_sources[data_source.upper()].prices[pair][date]["price"],
                            self.data_sources[data_source.upper()].assets[asset]["name"],
                            self.data_sources[data_source.upper()].prices[pair][date]["url"],
                        )

                # Esegue la richiesta per il prezzo storico
                self.data_sources[data_source.upper()].get_historical(asset, quote, timestamp)
                if (
                    pair in self.data_sources[data_source.upper()].prices
                    and date in self.data_sources[data_source.upper()].prices[pair]
                ):
                    return (
                        self.data_sources[data_source.upper()].prices[pair][date]["price"],
                        self.data_sources[data_source.upper()].assets[asset]["name"],
                        self.data_sources[data_source.upper()].prices[pair][date]["url"],
                    )

                # Se il prezzo non disponibile, memorizza la richiesta fallita
                print(f"Failed to retrieve price for {asset} on {date}. Marking request as failed.")
                self.failed_requests.add((data_source, asset, quote, date))
                self.save_failed_requests()  # Salva la richiesta fallita nel file
                return None, self.data_sources[data_source.upper()].assets[asset]["name"], SourceUrl("")

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
        for data_source in self.data_source_priority(asset):
            price, name, url = self.get_historical_ds(
                data_source, asset, quote, timestamp, no_cache
            )
            if price is not None:
                if config.debug:
                    print(
                        f"{Fore.YELLOW}price: {timestamp:%Y-%m-%d}, 1 "
                        f"{asset}={price.normalize():0,f} {quote} via "
                        f"{self.data_sources[data_source.upper()].name()} ({name})"
                    )
                if self.price_tool:
                    print(
                        f"{Fore.YELLOW}1 {asset}={price.normalize():0,f} {quote} "
                        f"{Fore.CYAN}via {self.data_sources[data_source.upper()].name()} ({name})"
                    )
                return price, name, self.data_sources[data_source.upper()].name(), url
        return None, name, DataSourceName(""), SourceUrl("")
