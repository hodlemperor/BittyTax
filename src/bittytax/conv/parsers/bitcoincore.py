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
    
    cryptoasset = kwargs.get("cryptoasset", "BTC")
    
    value = Decimal(row_dict["Importo (BTC)"])
    if value > 0:
        buy_quantity = value
        sell_quantity = None
        transaction_type = TrType.DEPOSIT
    else:
        buy_quantity = None
        sell_quantity = abs(value)  # Assicurati che il valore sia positivo
        transaction_type = TrType.WITHDRAWAL

    data_row.t_record = TransactionOutRecord(
        transaction_type,
        data_row.timestamp,
        buy_quantity=buy_quantity,
        sell_quantity=sell_quantity,
        buy_asset=cryptoasset if buy_quantity is not None else "",
        sell_asset=cryptoasset if sell_quantity is not None else "",
        fee_quantity=None,  # Aggiungi logica per la quantità della commissione se necessario
        fee_asset=cryptoasset if sell_quantity is not None else "",
        wallet=WALLET,
        note=row_dict.get("Etichetta", "")
    )

DataParser(
    ParserType.WALLET,
    "Bitcoin Wallet",
    ["Confermato", "Data", "Tipo", "Etichetta", "Indirizzo", "Importo (BTC)", "ID"],
    worksheet_name="Bitcoin Transactions",
    row_handler=parse_bitcoin_csv,
)
