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
    (3, 3): 0,
    # 0 is "no stress"
    # 1 is "primary stress": a mismatch is bad, but a match is good
    (0, 1): 1,
    (1, 0): 1,
    # 2 is "secondary stress": not as bad as a mismatch, but not as good as a match
    (0, 2): 0,
    (2, 0): 0,
    (1, 2): 0,
    (2, 1): 0,
    # 3 is a new line - very bad to match a syllable with a new line
    (0, 3): 100,
    (3, 0): 100,
    (1, 3): 100,
    (3, 1): 100,
    (2, 3): 100,
    (3, 2): 100,
    # missing syllable
    # match null char with song syllable
    (-1, 0): 1,
    (-1, 1): 1,
    (-1, 2): 1,
    (-1, 3): 0,
    # it's worse to skip a syllable in the passage than in the song
    # match passage syllable with null char
    (0, -1): 10,
    (1, -1): 10,
    (2, -1): 10,
    (3, -1): 0,
}


def dp_solve_stresses(passage_stresses, song_stresses, costs=default_costs):
    """
    Find the min-cost alignment of passage and song, using dynamic programming.
    * passage_stresses: array representation of passage stresses
    * song_stresses: array representation of song stresses
    """
    # n_p, n_s = len(passage_stresses), len(song_stresses)
    @functools.cache
    def helper(i_p, i_s):
        """
        Returns min-cost alignment of passage_stresses[:i_p+1] and song_stresses[:i_s+1]

        Result is (
            total cost,
            marginal cost,
            step,
            next_args
        )
        """
        if i_p < 0:
            temp_cost = 0
            for s_syllable in song_stresses[:i_s+1]:
                temp_cost += costs[(-1, s_syllable)]
            return (temp_cost, temp_cost, None, None)
            # return (0, 0, None, None)
        if i_s < 0:
            # trying to put a passage line before the first song line
            temp_cost = 0
            for p_syllable in passage_stresses[:i_p+1]:
                temp_cost += costs[(p_syllable, -1)]
            return (temp_cost, temp_cost, None, None)
        if passage_stresses[i_p] == song_stresses[i_s]:
            # characters match
            next_cost, _, _, _ = helper(i_p-1, i_s-1)
            this_cost = costs[(passage_stresses[i_p], song_stresses[i_s])]
            return (
                next_cost + this_cost,
                this_cost,
                (i_p, i_s), # match i_p with i_s
                (i_p-1, i_s-1) # next soln
            )
        if passage_stresses[i_p] != song_stresses[i_s]:
            # characters don't match
            if song_stresses[i_s] == 3:
                # new line in song - can only match with new line in passage
                # so let's skip syllables in passage, and hope the next syllable
                # in the passage is a new line
                next_cost, _, _, _ = helper(i_p-1, i_s)
                this_cost = costs[(passage_stresses[i_p], -1)]
                return (
                    next_cost + this_cost,
                    this_cost,
                    (i_p, -1), # match i_p with -1 (blank character)
                    (i_p-1, i_s) # next soln
                )
            if passage_stresses[i_p] == 3:
                # new line in passage - this is okay to skip!
                next_cost, _, _, _ = helper(i_p-1, i_s)
                this_cost = costs[(3, -1)]
                return (
                    next_cost + this_cost,
                    this_cost,
                    (i_p, -1), # match i_p with -1 (blank character)
                    (i_p-1, i_s) # next soln
                )
            # skip syllable in passage
            cost_skip_p, _, _, _ = helper(i_p-1, i_s)
            this_cost_skip_p = costs[(passage_stresses[i_p], -1)]
            cost_skip_p += this_cost_skip_p
            # skip syllable in song
            cost_skip_s, _, _, _ = helper(i_p, i_s-1)
            this_cost_skip_s = costs[(-1, song_stresses[i_s])]
            cost_skip_s += this_cost_skip_s
            # just match the syllables
            cost_match, _, _, _ = helper(i_p-1, i_s-1)
            this_cost_match = costs[(passage_stresses[i_p], song_stresses[i_s])]
            cost_match += this_cost_match
            return min(
                (cost_skip_p, this_cost_skip_p, (i_p, -1), (i_p-1, i_s)),
                (cost_skip_s, this_cost_skip_p, (-1, i_s), (i_p, i_s-1)),
                (cost_match, this_cost_match, (i_p, i_s), (i_p-1, i_s-1)),
                key=lambda x: x[0]
            )
    
    cost, marginal_cost, step, next_args = helper(len(passage_stresses) - 1, len(song_stresses) - 1)
    soln = []
    while step is not None:
        soln.append((*step, marginal_cost))
        _, marginal_cost, step, next_args = helper(*next_args)
    # reverse the solution - we built it backwards
    soln = soln[::-1]
    return cost, soln
        



# def dp_solve_stresses(passage_stresses, song_stresses, costs=default_costs):
#     """
#     Find the min-cost alignment of passage and song, using dynamic programming.
#     * passage_stresses: 2D nested array of lines, words, and syllables
#     * song_stresses: 2D nested array of lines and syllables
#     """
#     n_passage_lines = len(passage_stresses)
    
#     @functools.cache
#     def optimal_common_subsequence(p_start, p_end, s_index):
#         """
#         Find the optimal common subsequence of passage_line p_index and song_line s_index.
#         If the cost of skipping a syllable in p is the same as the cost of 
#         skipping a syllable in s, then this is the same as the longest common subsequence.

#         Represent a solution as a list of pairs (i, j) where i is the index of a
#         syllable in passage_line p_index and j is the index of a syllable in song_line s_index.
#         """
#         if s_index < 0:
#             # trying to put a passage line before the first song line
#             temp_cost = 0
#             for p_index in range(p_start, p_end):
#                 for p_syllable in passage_stresses[p_index]:
#                     temp_cost += costs[(p_syllable, -1)]
#             return (temp_cost, [])
#         if p_start == p_end:
#             # no passage lines to align
#             return (0, [])

#         n_s = len(song_stresses[s_index])
#         prev = [(0, []) for _ in range(n_s + 1)]
#         # for i_p in range(1, n_p + 1):
#         for p_index in range(p_start, p_end):
#             for i_p in range(1, len(passage_stresses[p_index]) + 1):
#                 # curr = [0] * (n_s + 1)
#                 curr = [(0, []) for _ in range(n_s + 1)]
#                 curr_p_syllable = passage_stresses[p_index][i_p-1]
#                 for i_s in range(1, n_s + 1):
#                     curr_s_syllable = song_stresses[s_index][i_s-1]
#                     curr[i_s] = min(
#                         # options:
#                         # 1. skip syllable in passage
#                         (
#                             costs[(-1, curr_p_syllable)] + prev[i_s][0],
#                             prev[i_s][1] + [(p_index, i_p-1, s_index, i_s-1)]
#                         ),
#                         # 2. skip syllable in song
#                         (
#                             costs[(curr_s_syllable, -1)] + curr[i_s-1][0],
#                             curr[i_s-1][1] + [(p_index, i_p-1, s_index, i_s-1)]
#                         ),
#                         # 3. match syllables
#                         (
#                             prev[i_s-1][0] + costs[(
#                                 curr_p_syllable, curr_s_syllable
#                             )],
#                             prev[i_s-1][1] + [(p_index, i_p-1, s_index, i_s-1)]
#                         ),
#                         key=lambda x: x[0]
#                     )
#                 prev = curr
#         return prev[-1]
    
#     # best_soln[(p_start, p_end, s_end)] = optimal (cost, solution)
#     # of aligning passage lines p_start through p_end with song
#     # up to and including song line s_end
#     best_soln = dict()
#     # best_soln[(0, 0, -1)] = (0, [], None)
#     for s_index in range(len(song_stresses)):
#         for p_start in range(n_passage_lines):
#             best_soln[(p_start, p_start, s_index-1)] = (0, [], None)
#             for p_end in range(p_start + 1, n_passage_lines + 1):
#                 # subproblem_cost, subproblem_soln = 0, []
#                 # if s_index > 0:
#                 if s_index <= 0:
#                     subproblem_cost, _ = optimal_common_subsequence(
#                         0, p_start, s_index - 1
#                     )
#                 else:
#                     subproblem_cost, _, _ = best_soln[(0, p_start, s_index-1)]
#                 # subproblem_cost, subproblem_soln = optimal_common_subsequence(
#                 #     0, p_start, s_index - 1
#                 # )
#                 cost, soln = optimal_common_subsequence(
#                     p_start, p_end, s_index, 
#                 )
#                 best_soln[(p_start, p_end, s_index)] = (
#                     subproblem_cost + cost,
#                     soln,
#                     (0, p_start, s_index-1) if s_index > 0 else None
#                 )
#     # reconstruct solution
#     final_cost, final_soln = 0, []

#     next_soln_key = min(
#         [k for k in best_soln.keys() 
#          if k[1] == n_passage_lines and k[2] == len(song_stresses) - 1],
#         key=lambda x: best_soln[x][0]
#     )



#     # next_soln_key = (0, n_passage_lines, len(song_stresses) - 1)
#     while next_soln_key is not None:
#         cost, soln, next_soln_key = best_soln[next_soln_key]
#         final_cost += cost
#         final_soln += soln
#     return final_cost, final_soln