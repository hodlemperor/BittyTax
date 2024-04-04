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

def parse_bitcoin_csv(
    data_row: "DataRow", _parser: DataParser, **kwargs: Unpack[ParserArgs]
) -> None:
    row_dict = data_row.row_dict
    data_row.timestamp = DataParser.parse_timestamp(row_dict["Data"], tz=config.local_timezone)
    
    # Assumi 'BTC' come cryptoasset se non specificato
    cryptoasset = kwargs.get("cryptoasset", "BTC")
    
    # Converte l'importo da stringa a Decimal
    value = Decimal(row_dict["Importo (BTC)"])
    
    # Determina il tipo di transazione e assicurati che buy_asset e sell_asset non siano mai None
    if value > 0:
        transaction_type = TrType.DEPOSIT
        buy_asset = cryptoasset
        sell_asset = ""  # Usa una stringa vuota invece di None
    else:
        transaction_type = TrType.WITHDRAWAL
        value = abs(value)  # Assicurati che il valore sia positivo per i prelievi
        buy_asset = ""
        sell_asset = cryptoasset

    # Nota: Se 'Etichetta' potrebbe essere None, considera anche di gestirla in modo simile
    note = row_dict["Etichetta"] if row_dict["Etichetta"] else ""

    data_row.t_record = TransactionOutRecord(
        transaction_type,
        data_row.timestamp,
        buy_quantity=value if transaction_type == TrType.DEPOSIT else None,
        sell_quantity=value if transaction_type == TrType.WITHDRAWAL else None,
        buy_asset=buy_asset,
        sell_asset=sell_asset,
        wallet=WALLET,
        note=note,
    )

# Esempio di registrazione del parser
DataParser(
    ParserType.WALLET,
    "Bitcoin Wallet",
    ["Confermato", "Data", "Tipo", "Etichetta", "Indirizzo", "Importo (BTC)", "ID"],
    worksheet_name="Bitcoin Transactions",
    row_handler=parse_bitcoin_csv,
)
