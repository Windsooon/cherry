from terminaltables import AsciiTable, DoubleTable, SingleTable

TABLE_DATA = (
    ('Platform', 'Years', 'Notes'),
    ('Mk5', '2007-2009', '你哈 Golf Mk5 Variant was\nintroduced in 2007.'),
    ('MKVI', '2009-2013', 'Might actually be Mk5.'),
)


def main():
    """Main function."""
    title = 'Confusion matrix'

    # AsciiTable.
    table_instance = AsciiTable(TABLE_DATA, title)
    table_instance.justify_columns[2] = 'right'
    print(table_instance.table)

if __name__ == '__main__':
    main()
