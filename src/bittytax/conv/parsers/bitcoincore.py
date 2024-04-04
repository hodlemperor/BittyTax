from decimal import Decimal
from typing import TYPE_CHECKING
from typing_extensions import Unpack

from ...bt_types import TrType
from ...config import config
from ..dataparser import DataParser, ParserArgs, ParserType
from ..exceptions import UnknownCryptoassetError
from ..out_record import TransactionOutRecord

if TYPE_CHECKING:
    from ..datarow import DataRow

WALLET = "Bitcoin Wallet"

def parse_bitcoin_csv(data_row: "DataRow", _parser: DataParser, **kwargs: Unpack[ParserArgs]) -> None:
    row_dict = data_row.row_dict
    # Converti il timestamp in un formato compatibile
    data_row.timestamp = DataParser.parse_timestamp(row_dict["Data"], tz=config.local_timezone)
    
    # Assumi 'BTC' come asset se non specificato
    cryptoasset = kwargs.get("cryptoasset", "BTC")
    
    # Converte l'importo da stringa a Decimal e determina la tipologia di transazione
    value = Decimal(row_dict["Importo (BTC)"])
    if value > 0:
        buy_asset = cryptoasset
        sell_asset = ""
        transaction_type = TrType.DEPOSIT
        buy_quantity = value
        sell_quantity = None
    else:
        buy_asset = ""
        sell_asset = cryptoasset
        transaction_type = TrType.WITHDRAWAL
        buy_quantity = None
        sell_quantity = abs(value)  # Converti in valore positivo per i prelievi
    
    # Costruisci l'oggetto TransactionOutRecord, assicurandoti che non ci siano valori None
    data_row.t_record = TransactionOutRecord(
        transaction_type=transaction_type,
        timestamp=data_row.timestamp,
        buy_quantity=buy_quantity,
        sell_quantity=sell_quantity,
        buy_asset=buy_asset,
        sell_asset=sell_asset,
        wallet=WALLET,
        note=row_dict.get("Etichetta", "")  # Usa get per evitare KeyError e fornire un valore di default
    )

# Esempio di registrazione del parser
DataParser(
    ParserType.WALLET,
    "Bitcoin Wallet",
    ["Confermato", "Data", "Tipo", "Etichetta", "Indirizzo", "Importo (BTC)", "ID"],
    worksheet_name="Bitcoin Transactions",
    row_handler=parse_bitcoin_csv,
)
