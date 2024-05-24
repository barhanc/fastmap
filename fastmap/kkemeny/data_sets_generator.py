import random


def generate_unique_vote(num_candidates):
    candidates = list(range(num_candidates))
    random.shuffle(candidates)
    return candidates


def generate_votes(num_voters, num_candidates):
    votes = []
    for _ in range(num_voters):
        votes.append(generate_unique_vote(num_candidates))
    return votes


if __name__ == "__main__":
    num_voters = 20
    num_candidates = 20
    votes = generate_votes(num_voters, num_candidates)
    print(votes)
