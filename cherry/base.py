def _get_stop_word(self):
    '''
    Stop word should store in the stop_words.dat file
    '''
    try:
        stop_word_path = os.path.join(DATA_DIR, 'stop_words.dat')
        with open(stop_word_path, encoding='utf-8') as f:
            stop_word = [l[:-1] for l in f.readlines()]
    except IOError:
        error = 'stop_words.dat not found'
        raise LanguageNotFoundError(error)
    return stop_word


def _tokenize(text):
    # for english:
    # return [
    #     t.lower() for t in nltp.word_tokenize(text) if len(t) > 1
    #     and t not in stop_word]
    return [
        t for t in jieba.cut(text, HMM=True) if len(t) > 1
        and t not in stop_word]
