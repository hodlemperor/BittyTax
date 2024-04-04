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
    # Converte il formato della data da ISO 8601 a timestamp UNIX, adattandolo al fuso orario locale
    data_row.timestamp = DataParser.parse_timestamp(row_dict["Data"], tz=config.local_timezone)
    
    # Assume 'BTC' come cryptoasset se non specificato
    cryptoasset = kwargs.get("cryptoasset", "BTC")
    
    # Converte il valore da stringa a Decimal
    value = Decimal(row_dict["Importo (BTC)"])
    
    # Determina il tipo di transazione basandosi sull'importo
    if value > 0:
        transaction_type = TrType.DEPOSIT
    else:
        transaction_type = TrType.WITHDRAWAL
        value = abs(value)  # Per le transazioni di tipo WITHDRAWAL, convertiamo il valore in positivo

    # Crea l'oggetto TransactionOutRecord con i dati pertinenti
    data_row.t_record = TransactionOutRecord(
        transaction_type,
        data_row.timestamp,
        buy_quantity=value if transaction_type == TrType.DEPOSIT else None,
        sell_quantity=value if transaction_type == TrType.WITHDRAWAL else None,
        buy_asset=cryptoasset if transaction_type == TrType.DEPOSIT else None,
        sell_asset=cryptoasset if transaction_type == TrType.WITHDRAWAL else None,
        wallet=WALLET,
        note=row_dict["Etichetta"],
    )

# Esempio di registrazione del parser
DataParser(
    ParserType.WALLET,
    "Bitcoin Wallet",
    ["Confermato", "Data", "Tipo", "Etichetta", "Indirizzo", "Importo (BTC)", "ID"],
    worksheet_name="Bitcoin Transactions",
    row_handler=parse_bitcoin_csv,
)
