from typing import List, Dict, Tuple


class SuffixAutomaton:
    def __init__(self):
        self.next: List[Dict[int, int]] = [{}]
        self.link: List[int] = [-1]
        self.length: List[int] = [0]
        self.last: int = 0
        self.occ_count: List[int] = [0]

    def extend(self, c: int):
        cur = len(self.next)
        self.next.append({})
        self.length.append(self.length[self.last] + 1)
        self.link.append(0)
        self.occ_count.append(1)

        p = self.last
        while p >= 0 and c not in self.next[p]:
            self.next[p][c] = cur
            p = self.link[p]

        if p == -1:
            self.link[cur] = 0
        else:
            q = self.next[p][c]
            if self.length[p] + 1 == self.length[q]:
                self.link[cur] = q
            else:
                clone = len(self.next)
                self.next.append(self.next[q].copy())
                self.length.append(self.length[p] + 1)
                self.link.append(self.link[q])
                self.occ_count.append(0)

                while p >= 0 and self.next[p].get(c) == q:
                    self.next[p][c] = clone
                    p = self.link[p]

                self.link[q] = self.link[cur] = clone

        self.last = cur

    def finalize_occurrences(self):
        order = sorted(range(len(self.length)), key=lambda i: self.length[i], reverse=True)
        for v in order:
            if self.link[v] != -1:
                self.occ_count[self.link[v]] += self.occ_count[v]

    def longest_repeated_substring(self, min_occ: int = 2) -> Tuple[int, int]:
        best_len = 0
        best_state = 0
        for i in range(len(self.length)):
            if self.occ_count[i] >= min_occ and self.length[i] > best_len:
                best_len = self.length[i]
                best_state = i
        return best_len, best_state
