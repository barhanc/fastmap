import random

class VotesGenerator:

    @staticmethod
    def generate_random_vote(num_candidates: int) -> list[int]:
        """Generates a single random vote."""
        candidates = list(range(num_candidates))
        random.shuffle(candidates)
        return candidates
    
    @staticmethod
    def generate_random_votes(num_candidates: int, num_votes: int) -> list[list[int]]:
        """Generates an num_candidates x num_votes matrix with each row being a permutation of numbers from 0 to num_votes-1."""
        matrix = []

        for _ in range(num_candidates):
            permutation = random.sample(range(num_votes), num_votes)
            matrix.append(permutation)

        return matrix
        