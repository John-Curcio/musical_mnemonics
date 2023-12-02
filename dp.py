import numpy as np
import cmudict
import re

class MnemonicFinder:

    def __init__(self, passage_text, song_text, aug_syllable_counts=None):
        self.passage_text = self.clean_text_keep_newlines_apostrophes(passage_text)
        print("Passage text:", self.passage_text)
        self.song_text = self.clean_text_keep_newlines_apostrophes(song_text)
        print("Song text:", self.song_text)
        if aug_syllable_counts is None:
            aug_syllable_counts = {}
        self.aug_syllable_counts = aug_syllable_counts
        self._cmudict = cmudict.dict()
        self.soln = None

    # @staticmethod
    # def clean_text(text):
    #     # Replace any non-alphabetical characters with a space
    #     cleaned_text = re.sub(r'[^a-zA-Z]+', ' ', text)
    #     return cleaned_text.strip()
    
    @staticmethod
    def clean_text_keep_newlines_apostrophes(text):
        # Replace any sequence of characters that are not letters, newlines, or apostrophes with a space
        cleaned_text = re.sub(r'[^\w\s\']', '', text)  # Keep word characters (letters, digits, underscores), whitespace, and apostrophes
        cleaned_text = re.sub(r'[\d_]+', ' ', cleaned_text)  # Remove digits and underscores
        cleaned_text = re.sub(r'[^\nA-Za-z\']+', ' ', cleaned_text)  # Replace other non-letter characters (except newlines and apostrophes) with space
        return cleaned_text.strip()

    def count_syllables(self, word):
        word = word.lower()
        if word not in self._cmudict:
            if word in self.aug_syllable_counts:
                return self.aug_syllable_counts[word]
            else:
                raise ValueError("Word not found in augmented cmudict:", word)

        # Count the number of syllables: each digit in the phonetic transcription represents a syllable
        # In the phonetic transcription, the stress of the syllable is indicated by a digit at the end of the syllable
        # For example, the word "abdomen" has 3 syllables, and its phonetic transcription is "AE0 B D AH1 M AH0 N"
        return sum([phoneme[-1].isdigit() for phoneme in self._cmudict[word][0]])
    
    def count_syllables_in_line(self, line):
        return sum([self.count_syllables(word) for word in line.split()])
    
    def align(self, max_lines=np.inf):
        passage_lines = self.passage_text.split('\n')
        song_lines = self.song_text.split('\n')
        passage_line_syll_counts = [self.count_syllables_in_line(word) for word in passage_lines]
        # print("Passage syllables:", passage_syllables)
        song_lines = [self.count_syllables_in_line(line) for line in song_lines]
        # print("Line syllables:", line_syllables)
        passage_line_breaks = align(passage_line_syll_counts, song_lines, max_lines=max_lines)
        passage_line_breaks = [0] + [x+1 for x in passage_line_breaks]
        print(passage_line_breaks)
        self.soln = [" ".join(passage_lines[passage_line_breaks[i]:passage_line_breaks[i+1]])
            for i in range(len(passage_line_breaks)-1)]
        return self.soln

    def display_soln(self):
        if self.soln is None:
            raise ValueError("No solution found yet")
        song_lines = self.song_text.split('\n')
        for i, line in enumerate(self.soln):
            soln_line_syllables = self.count_syllables_in_line(line)
            song_line_syallables = self.count_syllables_in_line(song_lines[i])
            syllable_diff = soln_line_syllables - song_line_syallables
            # display line and number of syllables to add or remove
            print(line, syllable_diff)

def align(passage_lines, song_lines, max_lines=np.inf):
    """
    How do we construct s_hat?
    There's a corresponding sequence breaks of length r-l+1
    s_hat[i] = sum(p[breaks[i]:breaks[i+1])
    
    actually, s has a sequence breaks too.
    so there's p_breaks and s_breaks
    

    Suppose it's
    * p: syllables in ith word of passage
    * s: syllables in ith word of song
    * s_breaks: words in ith line of song. words from s

    want p_breaks: words in ith line of song. words from p

    minimize sum(
    (
        sum(p[p_breaks[i]:p_breaks[i+1]]) - 
        sum(s[s_breaks[i]:s_breaks[i+1]])
    )^2
    for i in range(len(p_breaks)-1)
    )

    And let's always start the song at line 0

    Brute force: if you want K lines, then there are (W choose K)
        possible ways to break up the passage into lines. Because 
        you're assigning K words to be the last word in the line

    What if we choose the end of the song, first?
        Then if we're aligning suffixes with suffixes, we can recurse


    Suppose we have min_cost(w, n_lines)
    what's min_cost(w+1, n_lines)?

    min_cost(w+1, n_lines) = min(
        # put p[k:w+1] in last line of s, recurse
        min_cost(k, n_lines-1) + cost(p[k:w+1], s[n_lines])
        for k in range(w+1)
    )
    min_cost(w, 1) = cost(s[0:w+1], p[0])

    """
    max_lines = min(max_lines, len(song_lines))
    n_words = len(passage_lines)
    # passage_prefix_sum[i] = sum(passage_lines[:i])
    # passage_prefix_sums[0] = sum(passage_lines[:0]) = 0
    passage_prefix_sums = [0] + list(np.cumsum(passage_lines))
    
    def cost(passage_start, passage_end, line_index):
        """
        Returns (sum(p[passage_start:passage_end]) - s[song_line])**2
        sum(p[i:j]) = sum(p[:j]) - sum(p[:i])
        passage_prefix_sums[i] = sum(p[:i])
        """
        passage_line = passage_prefix_sums[passage_end] - passage_prefix_sums[passage_start]
        return (passage_line - song_lines[line_index])**2

    # start with just 1 line
    # represent a solution as a list of indices
    # solution[i] = index of last word in ith line
    prev = [
        ([w], cost(0, w+1, 0))
        for w in range(n_words)
    ]
    best_soln = prev[-1]
    for line_index in range(1, max_lines):
        # now that we've specified the number of lines our solution has,
        # let's try and cram the passage into that many lines

        curr = [(None, np.inf)] * n_words
        for w in range(n_words):
            # extend the words considered in the passage as much as we can
            for k in range(w+1):
                # words k:w+1 are in the last line
                curr_candidate = (
                    prev[k][0] + [w], # append index of last line to solution ending at k
                    prev[k][1] + cost(
                        passage_start=k+1,
                        passage_end=w+1, 
                        line_index=line_index)
                )
                curr[w] = min(
                    curr[w], 
                    curr_candidate,
                key=lambda x: x[1])

        best_soln = min(
            best_soln, 
            curr[-1],
        key=lambda x: x[1])
        prev = curr
    print("Best soln is", best_soln[0], "with cost", best_soln[1])
    return best_soln[0]

if __name__ == "__main__":

    test_passage = """
    The sieve of eratosthenes,
    the sieve of eratosthenes, 
    for i in range two square root n plus one,
    if i is prime then loop through j from i squared up to n plus one with step size i
    and mark j as not prime
    """
    test_song = """
    Look for the bare necessities
    The simple bare necessities
    Forget about your worries and your strife
    I mean the bare necessities
    Old Mother Nature's recipes
    That bring the bare necessities of life
    Wherever I wander, wherever I roam
    """
    aug_syllable_counts = {
        "eratosthenes": 5,
    }

    mnem = MnemonicFinder(test_passage, test_song, aug_syllable_counts)
    soln = mnem.align()
    print("\n-----\n".join(soln))