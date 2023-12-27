"""
Find min-cost alignment of a passage and a song, using dynamic programming.

"""

import functools
import numpy as np

default_costs = {
    # default_costs[(p, s)] = cost of matching passage syllable p with song syllable s
    (0, 0): 0,
    (1, 1): 0,
    (2, 2): 0,
    # 0 is "no stress"
    # 1 is "primary stress": a mismatch is bad, but a match is good
    (0, 1): 1,
    (1, 0): 1,
    # 2 is "secondary stress": not as bad as a mismatch, but not as good as a match
    (0, 2): 0,
    (2, 0): 0,
    (1, 2): 0,
    (2, 1): 0,
    # missing syllable
    # it's worse to skip a syllable in the passage than in the song
    (-1, 0): 10,
    (-1, 1): 10,
    (-1, 2): 10,
    (0, -1): 1,
    (1, -1): 1,
    (2, -1): 1,
}


def dp_solve_stresses(passage_stresses, song_stresses, costs=default_costs):
    """
    Find the min-cost alignment of passage and song, using dynamic programming.
    * passage_stresses: 2D nested array of lines, words, and syllables
    * song_stresses: 2D nested array of lines and syllables
    """
    n_passage_lines = len(passage_stresses)
    
    @functools.cache
    def optimal_common_subsequence(p_start, p_end, s_index):
        """
        Find the optimal common subsequence of passage_line p_index and song_line s_index.
        If the cost of skipping a syllable in p is the same as the cost of 
        skipping a syllable in s, then this is the same as the longest common subsequence.

        Represent a solution as a list of pairs (i, j) where i is the index of a
        syllable in passage_line p_index and j is the index of a syllable in song_line s_index.
        """
        if s_index < 0:
            # trying to put a passage line before the first song line
            temp_cost = 0
            for p_index in range(p_start, p_end):
                for p_syllable in passage_stresses[p_index]:
                    temp_cost += costs[(p_syllable, -1)]
            return (temp_cost, [])
        if p_start == p_end:
            # no passage lines to align
            return (0, [])

        n_s = len(song_stresses[s_index])
        prev = [(0, []) for _ in range(n_s + 1)]
        # for i_p in range(1, n_p + 1):
        for p_index in range(p_start, p_end):
            for i_p in range(1, len(passage_stresses[p_index]) + 1):
                # curr = [0] * (n_s + 1)
                curr = [(0, []) for _ in range(n_s + 1)]
                curr_p_syllable = passage_stresses[p_index][i_p-1]
                for i_s in range(1, n_s + 1):
                    curr_s_syllable = song_stresses[s_index][i_s-1]
                    curr[i_s] = min(
                        # options:
                        # 1. skip syllable in passage
                        (
                            costs[(-1, curr_p_syllable)] + prev[i_s][0],
                            prev[i_s][1] + [(p_index, i_p-1, s_index, i_s-1)]
                        ),
                        # 2. skip syllable in song
                        (
                            costs[(curr_s_syllable, -1)] + curr[i_s-1][0],
                            curr[i_s-1][1] + [(p_index, i_p-1, s_index, i_s-1)]
                        ),
                        # 3. match syllables
                        (
                            prev[i_s-1][0] + costs[(
                                curr_p_syllable, curr_s_syllable
                            )],
                            prev[i_s-1][1] + [(p_index, i_p-1, s_index, i_s-1)]
                        ),
                        key=lambda x: x[0]
                    )
                prev = curr
        return prev[-1]
    
    # best_soln[(p_start, p_end, s_end)] = optimal (cost, solution)
    # of aligning passage lines p_start through p_end with song
    # up to and including song line s_end
    best_soln = dict()
    # best_soln[(0, 0, -1)] = (0, [], None)
    for s_index in range(len(song_stresses)):
        for p_start in range(n_passage_lines):
            best_soln[(p_start, p_start, s_index-1)] = (0, [], None)
            for p_end in range(p_start + 1, n_passage_lines + 1):
                # subproblem_cost, subproblem_soln = 0, []
                # if s_index > 0:
                if s_index <= 0:
                    subproblem_cost, _ = optimal_common_subsequence(
                        0, p_start, s_index - 1
                    )
                else:
                    subproblem_cost, _, _ = best_soln[(0, p_start, s_index-1)]
                # subproblem_cost, subproblem_soln = optimal_common_subsequence(
                #     0, p_start, s_index - 1
                # )
                cost, soln = optimal_common_subsequence(
                    p_start, p_end, s_index, 
                )
                best_soln[(p_start, p_end, s_index)] = (
                    subproblem_cost + cost,
                    soln,
                    (0, p_start, s_index-1) if s_index > 0 else None
                )
    # reconstruct solution
    final_cost, final_soln = 0, []

    next_soln_key = min(
        [k for k in best_soln.keys() 
         if k[1] == n_passage_lines and k[2] == len(song_stresses) - 1],
        key=lambda x: best_soln[x][0]
    )



    # next_soln_key = (0, n_passage_lines, len(song_stresses) - 1)
    while next_soln_key is not None:
        cost, soln, next_soln_key = best_soln[next_soln_key]
        final_cost += cost
        final_soln += soln
    return final_cost, final_soln