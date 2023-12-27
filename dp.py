import numpy as np
import cmudict
import re
import os
from dp_syllables import dp_solve_stresses
from tqdm import tqdm
import json

def default_cost(diff):
    # it's better to be too short than too long
    if diff > 0:
        return 5 * diff**2
    else:
        return abs(diff)**2

class MultiSongMnemonicFinder:

    def __init__(self, passage_text, song_folder, aug_syllables=None, custom_cost=None):
        self.passage_text = SingleSongMnemonicFinder.clean_text_keep_newlines_apostrophes(passage_text)
        self.song_folder = song_folder
        if aug_syllables is None:
            aug_syllables = {}
        self.aug_syllables = aug_syllables
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

    def solve(self, use_stresses=False):
        # for song in songs, find the best alignment
        # return best alignment of all songs
        song_texts = self.get_songs()
        song_solns = []
        for song_name, song_text in tqdm(song_texts.items()):
            single_song_mnem = SingleSongMnemonicFinder(
                self.passage_text, 
                song_text, 
                self.aug_syllables,
                self.custom_cost,
                song_name,
            )
            if use_stresses:
                single_song_mnem.align_stresses()
            else:
                single_song_mnem.align()
            song_solns.append(single_song_mnem)
        song_solns = sorted(song_solns, key=lambda x: x.soln_cost)
        self.soln = song_solns[0]
        print("Solution cost:", self.soln.soln_cost)
        self.soln.display_soln()
        return self.soln



class SingleSongMnemonicFinder:

    def __init__(self, passage_text, song_text, aug_syllables=None, custom_cost=None, song_name=""):
        self.passage_text = self.clean_text_keep_newlines_apostrophes(passage_text)
        self.song_text = self.clean_text_keep_newlines_apostrophes(song_text)
        if aug_syllables is None:
            aug_syllables = {}
        if custom_cost is None:
            custom_cost = default_cost
        self.custom_cost = custom_cost
        self.song_name = song_name
        self.aug_syllables = aug_syllables
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
        if word in self.aug_syllables:
            return len(self.aug_syllables[word])
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
    
class StressMnemonicFinder(SingleSongMnemonicFinder):

    def __init__(self, passage_text, song_text, aug_syllables=None, custom_cost=None, song_name=""):
        super().__init__(passage_text, song_text, aug_syllables, custom_cost, song_name)
        self.passage_line_stresses = None
        self.song_line_stresses = None

    def extract_stresses_from_word(self, word):
        word = word.lower()
        if word in self.aug_syllables:
            return self.aug_syllables[word]
        if word not in self._cmudict:
            raise ValueError("Word not found in cmudict:", word)
        return [int(phoneme[-1]) for phoneme in self._cmudict[word][0]
                if phoneme[-1].isdigit()]
    
    def extract_stresses_from_line(self, line):
        return [stress for word in line.split() 
                for stress in self.extract_stresses_from_word(word)]
    
    def align_stresses(self):
        passage_lines = self.passage_text.split('\n')
        song_lines = self.song_text.split('\n')
        self.passage_line_stresses = [
            # [self.extract_stresses_from_word(word) for word in line.split()]
            self.extract_stresses_from_line(line)
            for line in passage_lines
        ]
        print(self.passage_line_stresses)
        self.song_line_stresses = [
            self.extract_stresses_from_line(line)
            for line in song_lines
        ]
        print(self.song_line_stresses)
        # soln takes the form of a list [
        #   (passage_line_index, passage_syllable_index, song_line_index, song_syllable_index)
        #]
        soln_cost, soln = dp_solve_stresses(
            self.passage_line_stresses, 
            self.song_line_stresses, 
        )
        self.soln = soln
        self.soln_cost = soln_cost
        print("Solution cost:", soln_cost)
        print("Solution:", soln)
        # TODO: this solution is very difficult to represent as text
        # for p_line, p_syllable, s_line, s_syllable in soln:
            


        # passage_line_breaks = [0] + [x+1 for x in passage_line_breaks]
        # self.soln = [" ".join(passage_lines[passage_line_breaks[i]:passage_line_breaks[i+1]])
        #     for i in range(len(passage_line_breaks)-1)]
        # self.soln_cost = soln_cost
        # return self.soln

    def display_soln(self):
        # soln takes the form of a list [
        #   (passage_line_index, passage_syllable_index, song_line_index, song_syllable_index)
        #]
        print("Solution cost:", self.soln_cost)
        print("Solution:", self.soln)

        passage_words = [line.split() for line in self.passage_text.split('\n')]
        
        p_word_index = 0
        for p_line, p_syllable, s_line, s_syllable in self.soln:
            
            print(f"Passage line {p_line}, syllable {p_syllable} matches song line {s_line}, syllable {s_syllable}")
            print(f"{self.passage_line_stresses[p_line][p_syllable]} {self.song_line_stresses[s_line][s_syllable]}")

    
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

    with open("passages/sieve.txt", 'r') as f:
        passage_text = f.read()
    aug_syllables_json = json.load(open("aug_syllables.json", 'r'))
    aug_syllables_dict = {row["word"]: row["stresses"] 
                          for row in aug_syllables_json["words"]}
    with open("songs/bare_necessities.txt", 'r') as f:
        song_text = f.read()
    single_song_mnem = StressMnemonicFinder(
        passage_text, song_text, aug_syllables_dict
    )
    single_song_mnem.align_stresses()
    single_song_mnem.display_soln()




    # # accept passage and song as command line arguments
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("passage_file", help="path to passage file")
    # # add flag for whether to use stresses or just count syllables, default to syllables
    # parser.add_argument("--stress", action="store_true", help="use stresses instead of syllables")
    # parser.add_argument("--song_file", help="path to song file")
    # # parser.add_argument("song_folder", help="path to folder containing songs")
    # args = parser.parse_args()

    # # read passage and song
    # with open(os.path.join("passages", args.passage_file), 'r') as f:
    #     passage_text = f.read()
    # # song_folder = args.song_folder
    # # song_folder = "songs"
    # aug_syllables_json = json.load(open("aug_syllables.json", 'r'))
    # aug_syllables_dict = {row["word"]: row["stresses"] 
    #                       for row in aug_syllables_json["words"]}

    # use_stresses = args.stress
    
    # if args.song_file is not None:
    #     # with open(os.path.join("songs", args.song_file), 'r') as f:
    #     with open(args.song_file, 'r') as f:
    #         song_text = f.read()
    #     # find best alignment
    #     single_song_mnem = SingleSongMnemonicFinder(
    #         passage_text, song_text, aug_syllables_dict
    #     )
    #     if use_stresses:
    #         single_song_mnem.align_stresses()
    #     else:
    #         single_song_mnem.align()
    #         single_song_mnem.display_soln()
    # else:
    #     # find best alignment
    #     mnem_finder = MultiSongMnemonicFinder(passage_text, song_folder="songs", aug_syllables_dict)
    #     mnem_finder.solve()
    #     # mnem_finder.display_soln()
