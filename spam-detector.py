import collections
import os

TRAINING_NUM = 300
TESTING_NUM = 200


def get_stop_words():
    """
        Stop words include most common words used in Persian language.
        These words can effect the result of detecting spam words.
    """
    with open("stop-words", "r", encoding="utf-8") as f:
        return f.read().split()


COMMON_WORDS = get_stop_words()


class Paths:
    HAM_TESTING_PATH = os.getcwd() + "\\emails\\hamtesting"
    HAM_TRAINING_PATH = os.getcwd() + "\\emails\\hamtraining"
    SPAM_TESTING_PATH = os.getcwd() + "\\emails\\spamtesting"
    SPAM_TRAINING_PATH = os.getcwd() + "\\emails\\spamtraining"


def get_words(path):
    with open(path, 'r', encoding='utf-8') as file:
        return [word for word in file.read().split() if word not in COMMON_WORDS]


def get_distribution(sample_path, k=500):

    distribution = collections.defaultdict(lambda: 1)
    files = os.listdir(sample_path)

    for f_name in files:
        list_of_words = get_words(sample_path + '/' + f_name)
        for word in list_of_words:
            distribution[word] += 1

    distribution = dict(sorted(distribution.items(), key=lambda x: x[1], reverse=True))

    return dict(list(distribution.items())[:k])


def get_spam_if_word(word, ham_distribution: dict, spam_distribution):
    """
    P(S|w) = P(w|S) / P(w|S) + P(w|H)
    P(w|S) = word_spam_count/total_spam_count
    P(w|H) = word_ham_count/total_ham_count

    """

    hams_total = sum(ham_distribution.values())
    spams_total = sum(spam_distribution.values())
    spam_most_frequents = spam_distribution.keys()
    ham_most_frequents = ham_distribution.keys()
    if (word in spam_most_frequents) and (word not in ham_most_frequents):
        return 0.999
    elif (word in ham_most_frequents) and (word not in spam_most_frequents):
        return 0.001
    elif word not in (spam_most_frequents or ham_most_frequents):
        return 0.5
    else:
        word_if_spam = spam_distribution.get(word, 0) / spams_total
        word_if_ham = (ham_distribution[word]) / hams_total
        return word_if_spam / float(word_if_spam + word_if_ham)


def train():
    return (get_distribution(Paths.HAM_TRAINING_PATH),
            get_distribution(Paths.SPAM_TRAINING_PATH),
            )


def is_spam(email_path: str, ham_distribution, spam_distribution):
    """To prevent very small probabilities (so division by zero)
    , top 200 words with very high or very low probabilities are chosen"""
    email_words = get_words(email_path)
    top_words = {}
    for word in email_words:
        top_words[word] = abs(get_spam_if_word(word, ham_distribution, spam_distribution) - 0.5)
    top_words = sorted(top_words, key=top_words.get, reverse=True)
    spam_if_email = 1
    ham_if_email = 1

    for word in top_words[:200]:
        p = get_spam_if_word(word, ham_distribution, spam_distribution)
        spam_if_email *= p
        ham_if_email *= (1 - p)
    if spam_if_email > ham_if_email:
        return True
    return False


def test_spams(ham_distribution, spam_distribution):
    results = []
    for i in range(1, TESTING_NUM + 1):
        results.append((i,
                        is_spam(f'{Paths.SPAM_TESTING_PATH}\\spamtesting ({i}).txt',
                                ham_distribution,
                                spam_distribution)))

    spam_count = len(list(filter(lambda x: x[1] is True, results)))
    return results, spam_count


def test_hams(ham_distribution, spam_distribution):
    results = []
    for i in range(1, TESTING_NUM + 1):
        results.append((i,
                        not is_spam(f'{Paths.HAM_TESTING_PATH}\\hamtesting ({i}).txt',
                                    ham_distribution,
                                    spam_distribution)))

    ham_count = len(list(filter(lambda x: x[1] is True, results)))
    return results, ham_count


if __name__ == '__main__':
    ham_distro, spam_distro = train()
    spam_results = test_spams(ham_distro, spam_distro)
    ham_results = test_hams(ham_distro, spam_distro)
    print("=" * 80)
    print("SPAM TEST RESULTS")
    print("=" * 80)

    for item in spam_results[0]:
        print(f'Test result {item[0]} = {"Correct" if item[1] else "Incorrect"}')
    print("=" * 80)
    print("HAM TEST RESULTS:")
    print("=" * 80)
    for item in ham_results[0]:
        print(f'Test result {item[0]} = {"Correct" if item[1] else "Incorrect"}')
    print("=" * 80)
    print(f'Spam ratio: {(spam_results[1] / TESTING_NUM) * 100}')
    print(f'Ham ratio: {(ham_results[1] / TESTING_NUM) * 100}')
