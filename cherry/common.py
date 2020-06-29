STOP_WORDS = {'Chinese': frozenset(['的'])}

BUILD_IN_MODELS = {
    'newsgroups': (
        'newsgroups.tar.gz',
        'https://raw.githubusercontent.com/Windsooon/cherry_datasets/master/newsgroups.tar.gz',
        '25952d9167b86d96503356d8272860d38d3929a31284bbb83d2737f50d23015e',
        'latin1',
        ),
    'review': (
        'review.tar.gz',
        'https://raw.githubusercontent.com/Windsooon/cherry_datasets/master/review.tar.gz',
        '9c46684a48054b1ffb6d74f0b0bfff7cda538c7e097f7a4f8e3d20b1f1e561db',
        'latin1',
        ),
    'email': (
        'email.tar.gz',
        'https://raw.githubusercontent.com/Windsooon/cherry_datasets/master/email.tar.gz',
        '901d7c5721ec4f72ad39bcf245c52c214a9d96ad96604ddaac86605bf9f910e2',
        'latin1',
        )
}
