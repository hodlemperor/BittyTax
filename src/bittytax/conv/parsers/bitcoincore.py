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

WALLET = "Bitcoin Core"


def parse_bitcoin_csv(
    data_row: "DataRow", _parser: DataParser, **kwargs: Unpack[ParserArgs]
) -> None:
    row_dict = data_row.row_dict
    # Adattamento per gestire il formato di data nel CSV
    data_row.timestamp = DataParser.parse_timestamp(row_dict["Data"], tz=config.local_timezone)

    # Gestione dell'importo e determinazione se è un deposito o un prelievo
    value = Decimal(row_dict["Importo (BTC)"])
    if value > 0:
        data_row.t_record = TransactionOutRecord(
            TrType.DEPOSIT,
            data_row.timestamp,
            buy_quantity=value,
            buy_asset="BTC",
            wallet=WALLET,
            note=row_dict["Etichetta"],
        )
    else:
        # Assumiamo che non ci siano commissioni separate nel file CSV fornito
        data_row.t_record = TransactionOutRecord(
            TrType.WITHDRAWAL,
            data_row.timestamp,
            sell_quantity=abs(value),
            sell_asset="BTC",
            wallet=WALLET,
            note=row_dict["Etichetta"],
        )

# Esempio di come registrare il parser nel sistema
DataParser(
    ParserType.WALLET,
    "Bitcoin Wallet",
    [
        "ID", "Etichetta", "Confermato", "Importo (BTC)", "Data"
    ],
    worksheet_name="Bitcoin Transactions",
    row_handler=parse_bitcoin_csv,
)
