import numpy as np
import cmudict
import re
import os
from tqdm import tqdm

def default_cost(diff):
    # it's better to be too short than too long
    if diff > 0:
        return 5 * diff**2
    else:
        return abs(diff)**2

class MultiSongMnemonicFinder:

    def __init__(self, passage_text, song_folder, aug_syllable_counts=None, custom_cost=None):
        self.passage_text = SingleSongMnemonicFinder.clean_text_keep_newlines_apostrophes(passage_text)
        self.song_folder = song_folder
        if aug_syllable_counts is None:
            aug_syllable_counts = {}
        self.aug_syllable_counts = aug_syllable_counts
        if custom_cost is None:
            custom_cost = default_cost
        self.custom_cost = custom_cost
        self._cmudict = cmudict.dict()
        self.soln = None

    def get_songs(self):
        # return list of song texts
        songs = {}
        for song in os.listdir(self.song_folder):
            with open(os.path.join(self.song_folder, song), 'r') as f:
                songs[song] = f.read()
        return songs

    def solve(self):
        # for song in songs, find the best alignment
        # return best alignment of all songs
        song_texts = self.get_songs()
        song_solns = []
        for song_name, song_text in tqdm(song_texts.items()):
            single_song_mnem = SingleSongMnemonicFinder(
                self.passage_text, 
                song_text, 
                self.aug_syllable_counts,
                self.custom_cost,
                song_name,
            )
            single_song_mnem.align()
            song_solns.append(single_song_mnem)
        song_solns = sorted(song_solns, key=lambda x: x.soln_cost)
        self.soln = song_solns[0]
        print("Solution cost:", self.soln.soln_cost)
        self.soln.display_soln()
        return self.soln



class SingleSongMnemonicFinder:

    def __init__(self, passage_text, song_text, aug_syllable_counts=None, custom_cost=None, song_name=""):
        self.passage_text = self.clean_text_keep_newlines_apostrophes(passage_text)
        self.song_text = self.clean_text_keep_newlines_apostrophes(song_text)
        if aug_syllable_counts is None:
            aug_syllable_counts = {}
        if custom_cost is None:
            custom_cost = default_cost
        self.custom_cost = custom_cost
        self.song_name = song_name
        self.aug_syllable_counts = aug_syllable_counts
        self._cmudict = cmudict.dict()
        self.soln = None
        self.soln_cost = None

    # @staticmethod
    # def clean_text(text):
    #     # Replace any non-alphabetical characters with a space
    #     cleaned_text = re.sub(r'[^a-zA-Z]+', ' ', text)
    #     return cleaned_text.strip()
    
    @staticmethod
    def clean_text_keep_newlines_apostrophes(text):
        # Replace any lines that begin with # with a space
        text = re.sub(r'^#.*$', ' ', text, flags=re.MULTILINE)
        # Replace any sequence of characters that are not letters, newlines, or apostrophes with a space
        cleaned_text = re.sub(r'[^\w\s\']', '', text)  # Keep word characters (letters, digits, underscores), whitespace, and apostrophes
        cleaned_text = re.sub(r'[\d_]+', ' ', cleaned_text)  # Remove digits and underscores
        cleaned_text = re.sub(r'[^\nA-Za-z\']+', ' ', cleaned_text)  # Replace other non-letter characters (except newlines and apostrophes) with space
        return cleaned_text.strip()

    def count_syllables(self, word):
        word = word.lower()
        if word in self.aug_syllable_counts:
            return self.aug_syllable_counts[word]
        if word not in self._cmudict:
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
        passage_line_breaks, soln_cost = self.dp_solve(passage_line_syll_counts, song_lines, max_lines=max_lines)
        passage_line_breaks = [0] + [x+1 for x in passage_line_breaks]
        self.soln = [" ".join(passage_lines[passage_line_breaks[i]:passage_line_breaks[i+1]])
            for i in range(len(passage_line_breaks)-1)]
        self.soln_cost = soln_cost
        return self.soln

    def display_soln(self):
        # print the song first
        # print("----Song----")
        # print(self.song_text)
        # print("----Passage----")
        if self.soln is None:
            raise ValueError("No solution found yet")
        print("Song:", self.song_name)
        # print cost of solution
        print("Solution cost:", self.soln_cost)
        soln_text = []
        song_lines = self.song_text.split('\n')
        for i, line in enumerate(self.soln):
            soln_line_syllables = self.count_syllables_in_line(line)
            song_line_syallables = self.count_syllables_in_line(song_lines[i])
            syllable_diff = soln_line_syllables - song_line_syallables
            # display line and number of syllables to add or remove
            syllable_string = ""
            if syllable_diff > 0:
                syllable_string = f"---- {syllable_diff} SYLLABLES TOO LONG"
            elif syllable_diff < 0:
                syllable_string = f"---- {-syllable_diff} TOO SHORT"
            # print(line, syllable_string)
            soln_text.append(line + syllable_string)
        print_passages_side_by_side(
            self.song_text, 
            "\n".join(soln_text), 
            title1="Song", 
            title2="Passage"
        )

    def dp_solve(self, passage_lines, song_lines, max_lines=np.inf, ):
        """
        How do we construct s_hat?

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
        # use this to efficiently calculate total number of syllables in a range
        passage_prefix_sums = [0] + list(np.cumsum(passage_lines))
        
        def cost(passage_start, passage_end, line_index):
            """
            Returns (sum(p[passage_start:passage_end]) - s[song_line])**2
            sum(p[i:j]) = sum(p[:j]) - sum(p[:i])
            passage_prefix_sums[i] = sum(p[:i])
            """
            passage_line = passage_prefix_sums[passage_end] - passage_prefix_sums[passage_start]
            # return (passage_line - song_lines[line_index])**2
            diff = passage_line - song_lines[line_index]
            return self.custom_cost(diff)
            # return abs(passage_line - song_lines[line_index])

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
        # print("Best soln is", best_soln[0], "with cost", best_soln[1])
        return best_soln
    
def print_passages_side_by_side(passage1, passage2, title1="Song", title2="Passage"):
    # Split each passage into lines
    lines1 = passage1.split('\n')
    lines2 = passage2.split('\n')

    # Ensure both lists have the same number of lines
    max_length = max(len(lines1), len(lines2))
    lines1.extend([''] * (max_length - len(lines1)))
    lines2.extend([''] * (max_length - len(lines2)))

    # Find the maximum length of a line in the first passage
    max_line_length = max(len(line) for line in lines1)

    # Print titles
    print(f"{title1.ljust(max_line_length)}    {title2}")

    # Print passages side by side
    for line1, line2 in zip(lines1, lines2):
        print(f"{line1.ljust(max_line_length)}    {line2}")



if __name__ == "__main__":

    # accept passage and song as command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("passage_file", help="path to passage file")
    parser.add_argument("--song_file", help="path to song file")
    # parser.add_argument("song_folder", help="path to folder containing songs")
    parser.add_argument("--aug_syllable_counts", help="path to file containing augmented syllable counts")
    args = parser.parse_args()

    # read passage and song
    with open(os.path.join("passages", args.passage_file), 'r') as f:
        passage_text = f.read()
    # song_folder = args.song_folder
    song_folder = "songs"
    # aug_syllable_counts = {}
    aug_syllable_counts = {
        "suffixes": 3,
        "oaken": 2,
        "worser": 2,
        "gon'": 1,
        "devotin'": 3,
        "floatin'": 2,
        "hakuna": 3,
        "matata": 3,
        "problemfree": 3,
        "huns": 1,
        "prefixes": 3,
        "newtold": 2,
        "derisions": 3,
        "our": 1,
        "every": 2,
        "rightmost": 2,
        "killashandra": 4,
        "num": 1,
        "sixtyfour": 3,
        "seargent": 2,
        "sergeant": 2,
        "hendersons": 3,
        "pablo": 2,
        "mr": 2,
        "nonzero": 3,
        "bolzano": 3,
        "weierstrass": 3,
        "cauchy": 2,
        "riemann": 2,

        "heapsort": 2,
        "heapify": 3,
        "def": 1,
        "nums": 1,
        "div": 1,
        "mod": 1,

        "heartbreakers": 3,
        "heartbreaker": 3,
        "caroling": 3,
        "rednosed": 2,
        "init": 2,
        "int": 1,
        "1": 1, "2": 1, "3": 1, "4": 1, "5": 1,
        "6": 1, "7": 2, "8": 1, "9": 1, "0": 2,
        }
    if args.aug_syllable_counts is not None:
        with open(args.aug_syllable_counts, 'r') as f:
            for line in f:
                word, count = line.split()
                aug_syllable_counts[word] = int(count)
    
    if args.song_file is not None:
        with open(os.path.join("songs", args.song_file), 'r') as f:
            song_text = f.read()
        # find best alignment
        single_song_mnem = SingleSongMnemonicFinder(
            passage_text, song_text, aug_syllable_counts
        )
        single_song_mnem.align()
        single_song_mnem.display_soln()
    else:
        # find best alignment
        mnem_finder = MultiSongMnemonicFinder(passage_text, song_folder, aug_syllable_counts)
        mnem_finder.solve()
        # mnem_finder.display_soln()
