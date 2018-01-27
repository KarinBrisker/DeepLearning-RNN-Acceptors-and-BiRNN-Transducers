import random

RANDOM_RANGE = 25


def generate_good_examples(num_examples, add_tag):
    good_examples = []
    for i in range(num_examples):
        list_rands = [None] * 9
        for j in range(9):
            list_rands[j] = random.randint(1, RANDOM_RANGE)

        good_ex = ''
        for k in range(list_rands[0]):
            good_ex += str(random.randint(1, 9))

        good_ex += 'a' * list_rands[1]

        for k in range(list_rands[2]):
            good_ex += str(random.randint(1, 9))

        good_ex += 'b' * list_rands[3]

        for k in range(list_rands[5]):
            good_ex += str(random.randint(1, 9))

        good_ex += 'c' * list_rands[6]

        for k in range(list_rands[7]):
            good_ex += str(random.randint(1, 9))

        good_ex += 'd' * list_rands[8]
        if add_tag:
            good_examples.append('good ' + good_ex)
        else:
            good_examples.append(good_ex)

    return good_examples


def generate_bad_examples(num_examples, add_tag):
    bad_examples = []
    for i in range(num_examples):
        list_rands = [None] * 9
        for j in range(9):
            list_rands[j] = random.randint(1, RANDOM_RANGE)

        bad_ex = ''
        for k in range(list_rands[0]):
            bad_ex += str(random.randint(1, 9))

        bad_ex += 'a' * list_rands[1]

        for k in range(list_rands[2]):
            bad_ex += str(random.randint(1, 9))

        bad_ex += 'c' * list_rands[3]

        for k in range(list_rands[5]):
            bad_ex += str(random.randint(1, 9))

        bad_ex += 'b' * list_rands[6]

        for k in range(list_rands[7]):
            bad_ex += str(random.randint(1, 9))

        bad_ex += 'd' * list_rands[8]
        if add_tag:
            bad_examples.append('bad ' + bad_ex)
        else:
            bad_examples.append(bad_ex)

    return bad_examples

def generate_train():

    pos_examples = generate_good_examples(4000, True)
    neg_examples = generate_bad_examples(4000, True)

    all_examples = pos_examples + neg_examples
    random.shuffle(all_examples)
    return all_examples



def generate_test():

    pos_examples = generate_good_examples(400, True)
    neg_examples = generate_bad_examples(400, True)

    all_examples = pos_examples + neg_examples
    all_examples = random.shuffle(all_examples)
    return all_examples


def generate_examples():
    pos_examples = generate_good_examples(500, False)
    neg_examples = generate_bad_examples(500, False)

    with open('pos_examples', 'w') as file1:
        for pos_example in pos_examples:
            file1.write("{0}\n".format(pos_example))

    with open('neg_examples', 'w') as file2:
        for neg_example in neg_examples:
            file2.write("{0}\n".format(neg_example))


if __name__ == '__main__':
    generate_examples()
    train = generate_train()
    test = generate_test()
