import cherry
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument(
    '-l', '--language', type=str, default='Chinese',
    help='Which language\'s dataset we use to test')
parser.add_argument(
    '-s', '--split', type=str, default=None,
    help='Split function to tokenizer data')
parser.add_argument(
    '-t', '--test_time', type=int, default=5,
    help='How many times we split data for testing')
parser.add_argument(
    '-n', '--num', type=int, default=60,
    help='How many test data we need every time')
parser.add_argument(
    '-d', dest='debug', const=True, action='store_const',
    help='Show wrong classified data')
args = parser.parse_args()


def main():
    print('This may takes some time, Go get a coffee :D.')
    a = cherry.analysis(
        lan=args.language, test_time=args.test_time,
        test_num=args.num, split=args.split, debug=args.debug)
    print(a.ctable)


if __name__ == '__main__':
    main()
