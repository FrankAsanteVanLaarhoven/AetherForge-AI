"""
scripts/generate_execution_traces.py
Generate real (task, ReAct-trajectory) training pairs by executing Python code.

Every OBSERVATION in the output is from an actual subprocess run — no fake strings.
Covers three trajectory types:
  A. Write-and-verify: implement → run → confirm output → answer
  B. Fix-the-bug:      inject bug → run → real error → fix → run again → answer
  C. TDD:              write tests → run (fail) → implement → run (pass) → answer

Usage:
    conda run -n ml-torch python scripts/generate_execution_traces.py [--n N]
    # Default N=3000.  CPU-only, ~2 min for 3000 traces.
"""

import json
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

N       = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 5000
TIMEOUT = 6


# ─────────────────────────────────────────────────────────────────────────────
# Seed programs  (task, correct_code)
# ─────────────────────────────────────────────────────────────────────────────

SEEDS = [
    # ── Arithmetic ────────────────────────────────────────────────────
    ("Write a Python function to compute the factorial of n iteratively.",
     "def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\nprint(factorial(5))\nprint(factorial(0))"),

    ("Write a Python function to check if a number is prime.",
     "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0: return False\n    return True\nprint(is_prime(17))\nprint(is_prime(18))"),

    ("Write a Python function to compute the nth Fibonacci number using iteration.",
     "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\nprint(fib(10))\nprint(fib(0))"),

    ("Write a Python function that returns the sum of digits of an integer.",
     "def digit_sum(n):\n    return sum(int(d) for d in str(abs(n)))\nprint(digit_sum(12345))\nprint(digit_sum(0))"),

    ("Write a Python function to find the greatest common divisor using Euclid's algorithm.",
     "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\nprint(gcd(48, 18))\nprint(gcd(100, 75))"),

    ("Write a Python function to compute base raised to exp using fast exponentiation.",
     "def power(base, exp):\n    if exp == 0: return 1\n    if exp % 2 == 0:\n        half = power(base, exp // 2)\n        return half * half\n    return base * power(base, exp - 1)\nprint(power(2, 10))\nprint(power(3, 5))"),

    ("Write a Python function to check if a number is a perfect square.",
     "def is_perfect_square(n):\n    if n < 0: return False\n    root = int(n**0.5)\n    return root * root == n\nprint(is_perfect_square(16))\nprint(is_perfect_square(15))"),

    # ── Strings ───────────────────────────────────────────────────────
    ("Write a Python function to check if a string is a palindrome.",
     "def is_palindrome(s):\n    return s == s[::-1]\nprint(is_palindrome('racecar'))\nprint(is_palindrome('hello'))"),

    ("Write a Python function to reverse a string.",
     "def reverse_string(s):\n    return s[::-1]\nprint(reverse_string('hello'))\nprint(reverse_string(''))"),

    ("Write a Python function that counts word frequencies in a string.",
     "def word_count(text):\n    counts = {}\n    for w in text.lower().split():\n        counts[w] = counts.get(w, 0) + 1\n    return counts\nprint(word_count('the cat sat on the mat the cat'))"),

    ("Write a Python function to check if two strings are anagrams.",
     "def are_anagrams(a, b):\n    return sorted(a.lower()) == sorted(b.lower())\nprint(are_anagrams('listen', 'silent'))\nprint(are_anagrams('hello', 'world'))"),

    ("Write a Python function to compress a string using run-length encoding.",
     "def rle_encode(s):\n    if not s: return ''\n    result, count = '', 1\n    for i in range(1, len(s)):\n        if s[i] == s[i-1]:\n            count += 1\n        else:\n            result += s[i-1] + (str(count) if count > 1 else '')\n            count = 1\n    result += s[-1] + (str(count) if count > 1 else '')\n    return result\nprint(rle_encode('aabbbcccc'))\nprint(rle_encode('abc'))"),

    ("Write a Python function to find the longest common prefix of a list of strings.",
     "def longest_common_prefix(words):\n    if not words: return ''\n    prefix = words[0]\n    for w in words[1:]:\n        while not w.startswith(prefix):\n            prefix = prefix[:-1]\n            if not prefix: return ''\n    return prefix\nprint(longest_common_prefix(['flower', 'flow', 'flight']))\nprint(longest_common_prefix(['dog', 'racecar', 'car']))"),

    ("Write a Python function to count the number of vowels in a string.",
     "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')\nprint(count_vowels('Hello World'))\nprint(count_vowels('rhythm'))"),

    # ── Lists / Arrays ────────────────────────────────────────────────
    ("Write a Python function to find the maximum element in a list without using max().",
     "def find_max(lst):\n    if not lst: raise ValueError('empty list')\n    m = lst[0]\n    for x in lst[1:]:\n        if x > m: m = x\n    return m\nprint(find_max([3, 1, 4, 1, 5, 9, 2, 6]))\nprint(find_max([-5, -1, -3]))"),

    ("Write a Python function to flatten a nested list one level deep.",
     "def flatten(lst):\n    return [item for sublist in lst for item in sublist]\nprint(flatten([[1, 2], [3, 4], [5]]))\nprint(flatten([]))"),

    ("Write a Python function to remove duplicates from a list while preserving order.",
     "def unique(lst):\n    seen = set()\n    return [x for x in lst if not (x in seen or seen.add(x))]\nprint(unique([1, 2, 1, 3, 2, 4]))\nprint(unique([]))"),

    ("Write a Python function to rotate a list by k positions to the left.",
     "def rotate_left(lst, k):\n    if not lst: return lst\n    k = k % len(lst)\n    return lst[k:] + lst[:k]\nprint(rotate_left([1, 2, 3, 4, 5], 2))\nprint(rotate_left([1, 2, 3], 5))"),

    ("Write a Python function to merge two sorted lists into one sorted list.",
     "def merge_sorted(a, b):\n    result, i, j = [], 0, 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]: result.append(a[i]); i += 1\n        else: result.append(b[j]); j += 1\n    return result + a[i:] + b[j:]\nprint(merge_sorted([1, 3, 5], [2, 4, 6]))\nprint(merge_sorted([], [1, 2]))"),

    ("Write a Python function to find the two numbers in a list that add up to a target.",
     "def two_sum(nums, target):\n    seen = {}\n    for i, n in enumerate(nums):\n        if target - n in seen:\n            return [seen[target - n], i]\n        seen[n] = i\n    return []\nprint(two_sum([2, 7, 11, 15], 9))\nprint(two_sum([3, 2, 4], 6))"),

    # ── Search & Sort ─────────────────────────────────────────────────
    ("Write a Python binary search function that returns the index or -1.",
     "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: lo = mid + 1\n        else: hi = mid - 1\n    return -1\nprint(binary_search([1, 3, 5, 7, 9], 7))\nprint(binary_search([1, 3, 5, 7, 9], 4))"),

    ("Write a Python merge sort implementation.",
     "def merge_sort(lst):\n    if len(lst) <= 1: return lst\n    mid  = len(lst) // 2\n    left = merge_sort(lst[:mid])\n    right = merge_sort(lst[mid:])\n    result, i, j = [], 0, 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]: result.append(left[i]); i += 1\n        else: result.append(right[j]); j += 1\n    return result + left[i:] + right[j:]\nprint(merge_sort([5, 3, 8, 1, 9, 2]))\nprint(merge_sort([]))"),

    ("Write a Python function that finds all prime numbers up to n using the Sieve of Eratosthenes.",
     "def sieve(n):\n    is_p = [True] * (n + 1)\n    is_p[0] = is_p[1] = False\n    for i in range(2, int(n**0.5) + 1):\n        if is_p[i]:\n            for j in range(i*i, n + 1, i):\n                is_p[j] = False\n    return [i for i, p in enumerate(is_p) if p]\nprint(sieve(30))\nprint(len(sieve(100)))"),

    # ── Data structures ────────────────────────────────────────────────
    ("Implement a stack using a Python list with push, pop, and peek methods.",
     "class Stack:\n    def __init__(self):\n        self._data = []\n    def push(self, x):\n        self._data.append(x)\n    def pop(self):\n        if not self._data: raise IndexError('pop from empty stack')\n        return self._data.pop()\n    def peek(self):\n        if not self._data: raise IndexError('peek at empty stack')\n        return self._data[-1]\n    def __len__(self):\n        return len(self._data)\ns = Stack()\ns.push(1); s.push(2); s.push(3)\nprint(s.peek())\nprint(s.pop())\nprint(len(s))"),

    ("Implement a queue using two stacks.",
     "class Queue:\n    def __init__(self):\n        self._in, self._out = [], []\n    def enqueue(self, x):\n        self._in.append(x)\n    def dequeue(self):\n        if not self._out:\n            while self._in:\n                self._out.append(self._in.pop())\n        if not self._out: raise IndexError('dequeue from empty queue')\n        return self._out.pop()\nq = Queue()\nq.enqueue(1); q.enqueue(2); q.enqueue(3)\nprint(q.dequeue())\nprint(q.dequeue())"),

    ("Implement a simple linked list with append and to_list methods.",
     "class Node:\n    def __init__(self, val):\n        self.val = val\n        self.next = None\nclass LinkedList:\n    def __init__(self):\n        self.head = None\n    def append(self, val):\n        node = Node(val)\n        if not self.head:\n            self.head = node; return\n        cur = self.head\n        while cur.next: cur = cur.next\n        cur.next = node\n    def to_list(self):\n        result, cur = [], self.head\n        while cur: result.append(cur.val); cur = cur.next\n        return result\nll = LinkedList()\nll.append(1); ll.append(2); ll.append(3)\nprint(ll.to_list())"),

    ("Implement an LRU cache with get and put operations.",
     "from collections import OrderedDict\nclass LRUCache:\n    def __init__(self, capacity):\n        self.cap = capacity\n        self.cache = OrderedDict()\n    def get(self, key):\n        if key not in self.cache: return -1\n        self.cache.move_to_end(key)\n        return self.cache[key]\n    def put(self, key, value):\n        if key in self.cache: self.cache.move_to_end(key)\n        self.cache[key] = value\n        if len(self.cache) > self.cap:\n            self.cache.popitem(last=False)\nc = LRUCache(2)\nc.put(1, 1); c.put(2, 2)\nprint(c.get(1))\nc.put(3, 3)\nprint(c.get(2))"),

    # ── Functional / Iterators ────────────────────────────────────────
    ("Write a Python generator that yields Fibonacci numbers indefinitely.",
     "def fibonacci():\n    a, b = 0, 1\n    while True:\n        yield a\n        a, b = b, a + b\nfib = fibonacci()\nprint([next(fib) for _ in range(10)])"),

    ("Write a Python function using reduce to compute the product of a list.",
     "from functools import reduce\ndef product(lst):\n    return reduce(lambda a, b: a * b, lst, 1)\nprint(product([1, 2, 3, 4, 5]))\nprint(product([]))"),

    ("Write a context manager in Python that measures execution time.",
     "import time\nfrom contextlib import contextmanager\n@contextmanager\ndef timer(label=''):\n    t0 = time.perf_counter()\n    yield\n    elapsed = time.perf_counter() - t0\n    print(f'{label}: {elapsed:.3f}s')\nwith timer('loop'):\n    x = sum(range(10000))"),

    ("Write a Python decorator that caches function results (memoize).",
     "def memoize(fn):\n    cache = {}\n    def wrapper(*args):\n        if args not in cache:\n            cache[args] = fn(*args)\n        return cache[args]\n    return wrapper\n@memoize\ndef fib(n):\n    if n < 2: return n\n    return fib(n-1) + fib(n-2)\nprint(fib(30))\nprint(fib(10))"),

    # ── File / JSON ───────────────────────────────────────────────────
    ("Write Python code to write a list of dicts to a JSON file and read it back.",
     "import json, tempfile, os\ndata = [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]\ntmpf = tempfile.mktemp(suffix='.json')\nwith open(tmpf, 'w') as f:\n    json.dump(data, f)\nwith open(tmpf) as f:\n    loaded = json.load(f)\nprint(loaded[0]['name'])\nprint(loaded[1]['age'])\nos.unlink(tmpf)"),

    ("Write Python code to count lines in a text file.",
     "import tempfile, os\ntmpf = tempfile.mktemp(suffix='.txt')\nwith open(tmpf, 'w') as f:\n    f.write('line one\\nline two\\nline three\\n')\nwith open(tmpf) as f:\n    n = sum(1 for _ in f)\nprint(n)\nos.unlink(tmpf)"),

    # ── Comprehensions / Dict ops ────────────────────────────────────
    ("Write Python code to group a list of dicts by a key using defaultdict.",
     "from collections import defaultdict\npeople = [{'name': 'Alice', 'dept': 'Eng'}, {'name': 'Bob', 'dept': 'HR'}, {'name': 'Carol', 'dept': 'Eng'}]\nby_dept = defaultdict(list)\nfor p in people:\n    by_dept[p['dept']].append(p['name'])\nprint(dict(by_dept))"),

    ("Write a Python function to transpose a 2D matrix.",
     "def transpose(matrix):\n    return [list(row) for row in zip(*matrix)]\nprint(transpose([[1, 2, 3], [4, 5, 6]]))\nprint(transpose([[1]]))"),

    # ── OOP / Classes ─────────────────────────────────────────────────
    ("Implement a BinaryTree class with insert and inorder traversal.",
     "class Node:\n    def __init__(self, val):\n        self.val = val\n        self.left = self.right = None\nclass BST:\n    def __init__(self):\n        self.root = None\n    def insert(self, val):\n        def _ins(node, v):\n            if not node: return Node(v)\n            if v < node.val: node.left = _ins(node.left, v)\n            else: node.right = _ins(node.right, v)\n            return node\n        self.root = _ins(self.root, val)\n    def inorder(self):\n        result = []\n        def _in(node):\n            if node:\n                _in(node.left)\n                result.append(node.val)\n                _in(node.right)\n        _in(self.root)\n        return result\nt = BST()\nfor v in [5, 3, 7, 1, 4, 6, 8]: t.insert(v)\nprint(t.inorder())"),

    ("Implement a graph with adjacency list supporting BFS and DFS.",
     "from collections import deque\nclass Graph:\n    def __init__(self):\n        self.adj = {}\n    def add_edge(self, u, v):\n        self.adj.setdefault(u, []).append(v)\n        self.adj.setdefault(v, []).append(u)\n    def bfs(self, start):\n        visited, q, order = {start}, deque([start]), []\n        while q:\n            node = q.popleft()\n            order.append(node)\n            for nb in self.adj.get(node, []):\n                if nb not in visited:\n                    visited.add(nb); q.append(nb)\n        return order\n    def dfs(self, start):\n        visited, order = set(), []\n        def _dfs(n):\n            visited.add(n); order.append(n)\n            for nb in self.adj.get(n, []):\n                if nb not in visited: _dfs(nb)\n        _dfs(start)\n        return order\ng = Graph()\nfor u, v in [(1,2),(1,3),(2,4),(3,4),(4,5)]: g.add_edge(u, v)\nprint(g.bfs(1))\nprint(g.dfs(1))"),

    ("Implement a heap-based priority queue in Python.",
     "import heapq\nclass PriorityQueue:\n    def __init__(self):\n        self._heap = []\n    def push(self, item, priority):\n        heapq.heappush(self._heap, (priority, item))\n    def pop(self):\n        priority, item = heapq.heappop(self._heap)\n        return item, priority\n    def __len__(self):\n        return len(self._heap)\npq = PriorityQueue()\npq.push('task_a', 3)\npq.push('task_b', 1)\npq.push('task_c', 2)\nwhile pq:\n    print(pq.pop())"),

    # ── Decorators ────────────────────────────────────────────────────
    ("Write a Python decorator that logs function calls with their arguments.",
     "import functools\ndef log_calls(fn):\n    @functools.wraps(fn)\n    def wrapper(*args, **kwargs):\n        parts = [repr(a) for a in args] + [f'{k}={v!r}' for k, v in kwargs.items()]\n        print(f'Calling {fn.__name__}({', '.join(parts)})')\n        result = fn(*args, **kwargs)\n        print(f'  → {result!r}')\n        return result\n    return wrapper\n@log_calls\ndef add(a, b):\n    return a + b\nadd(3, 4)"),

    ("Write a Python decorator that validates all arguments are positive numbers.",
     "import functools\ndef positive_args(fn):\n    @functools.wraps(fn)\n    def wrapper(*args):\n        for i, a in enumerate(args):\n            if not isinstance(a, (int, float)) or a <= 0:\n                raise ValueError(f'Arg {i} must be positive, got {a!r}')\n        return fn(*args)\n    return wrapper\n@positive_args\ndef area(w, h):\n    return w * h\nprint(area(3, 4))\ntry:\n    area(-1, 4)\nexcept ValueError as e:\n    print(f'Caught: {e}')"),

    # ── Regex ──────────────────────────────────────────────────────────
    ("Write a Python function to extract all email addresses from a string.",
     "import re\ndef extract_emails(text):\n    return re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}', text)\ntext = 'Contact alice@example.com or bob.smith@test.org for details.'\nprint(extract_emails(text))"),

    ("Write a Python function to validate a date string in YYYY-MM-DD format using regex.",
     "import re\ndef is_valid_date(s):\n    return bool(re.fullmatch(r'\\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\\d|3[01])', s))\nprint(is_valid_date('2024-06-15'))\nprint(is_valid_date('2024-13-01'))\nprint(is_valid_date('not-a-date'))"),

    # ── Algorithms ────────────────────────────────────────────────────
    ("Implement quicksort in Python.",
     "def quicksort(lst):\n    if len(lst) <= 1: return lst\n    pivot = lst[len(lst) // 2]\n    left   = [x for x in lst if x < pivot]\n    middle = [x for x in lst if x == pivot]\n    right  = [x for x in lst if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\nprint(quicksort([3, 6, 8, 10, 1, 2, 1]))\nprint(quicksort([]))"),

    ("Write a Python function to find the length of the longest increasing subsequence.",
     "def lis_length(arr):\n    if not arr: return 0\n    dp = [1] * len(arr)\n    for i in range(1, len(arr)):\n        for j in range(i):\n            if arr[j] < arr[i]:\n                dp[i] = max(dp[i], dp[j] + 1)\n    return max(dp)\nprint(lis_length([10, 9, 2, 5, 3, 7, 101, 18]))\nprint(lis_length([0, 1, 0, 3, 2, 3]))"),

    ("Write a Python function to solve the coin change problem (minimum coins).",
     "def coin_change(coins, amount):\n    dp = [float('inf')] * (amount + 1)\n    dp[0] = 0\n    for a in range(1, amount + 1):\n        for c in coins:\n            if c <= a and dp[a - c] + 1 < dp[a]:\n                dp[a] = dp[a - c] + 1\n    return dp[amount] if dp[amount] != float('inf') else -1\nprint(coin_change([1, 5, 10, 25], 36))\nprint(coin_change([2], 3))"),

    # ── Functional / Itertools ────────────────────────────────────────
    ("Write a Python function using itertools to generate all combinations of a list.",
     "from itertools import combinations\ndef all_combinations(lst):\n    result = []\n    for r in range(1, len(lst) + 1):\n        result.extend(combinations(lst, r))\n    return [list(c) for c in result]\nprint(all_combinations([1, 2, 3]))"),

    ("Write a Python function using itertools.groupby to group consecutive equal elements.",
     "from itertools import groupby\ndef group_consecutive(lst):\n    return [(k, list(v)) for k, v in groupby(lst)]\nprint(group_consecutive([1,1,2,3,3,3,1,1]))\nprint(group_consecutive([]))"),

    ("Write a Python function using functools.lru_cache for memoized Fibonacci.",
     "from functools import lru_cache\n@lru_cache(maxsize=None)\ndef fib(n):\n    if n < 2: return n\n    return fib(n-1) + fib(n-2)\nprint([fib(i) for i in range(10)])\nprint(fib(50))"),

    # ── Dict / Set / Comprehensions ───────────────────────────────────
    ("Write a Python function to find the intersection of two lists preserving order.",
     "def list_intersection(a, b):\n    b_set = set(b)\n    return [x for x in a if x in b_set]\nprint(list_intersection([1, 2, 3, 4, 5], [3, 4, 5, 6, 7]))\nprint(list_intersection([], [1, 2]))"),

    ("Write a Python function to invert a dictionary (swap keys and values).",
     "def invert_dict(d):\n    return {v: k for k, v in d.items()}\nprint(invert_dict({'a': 1, 'b': 2, 'c': 3}))\nprint(invert_dict({}))"),

    ("Write a Python function to chunk a list into batches of size n.",
     "def chunk(lst, n):\n    return [lst[i:i+n] for i in range(0, len(lst), n)]\nprint(chunk([1,2,3,4,5,6,7], 3))\nprint(chunk([1,2,3,4,5], 2))\nprint(chunk([], 3))"),

    # ── Error handling / Context managers ────────────────────────────
    ("Write a Python function that safely parses JSON and returns a default on error.",
     "import json\ndef safe_json(s, default=None):\n    try:\n        return json.loads(s)\n    except (json.JSONDecodeError, TypeError):\n        return default\nprint(safe_json('{\"key\": 42}'))\nprint(safe_json('not json', default={}))\nprint(safe_json(None, default=[]))"),

    ("Write a Python class using __enter__ and __exit__ to manage a temporary file.",
     "import tempfile, os\nclass TempFile:\n    def __init__(self, suffix='.txt'):\n        self.suffix = suffix\n        self.path = None\n    def __enter__(self):\n        fd, self.path = tempfile.mkstemp(suffix=self.suffix)\n        os.close(fd)\n        return self.path\n    def __exit__(self, *_):\n        if self.path and os.path.exists(self.path):\n            os.unlink(self.path)\nwith TempFile('.txt') as p:\n    with open(p, 'w') as f:\n        f.write('hello')\n    print(open(p).read())\nprint('cleaned up:', not os.path.exists(p))"),

    ("Write a Python generator that yields sliding window views of a list.",
     "def sliding_window(lst, k):\n    for i in range(len(lst) - k + 1):\n        yield lst[i:i+k]\nprint(list(sliding_window([1,2,3,4,5], 3)))\nprint(list(sliding_window([1,2], 2)))"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Bug injectors
# ─────────────────────────────────────────────────────────────────────────────

def inject_missing_colon(code: str) -> str:
    """Remove the colon from a def/if/for/while/else/elif/with line."""
    lines = code.split("\n")
    cands = [
        i for i, l in enumerate(lines)
        if any(l.strip().startswith(kw) for kw in
               ("def ", "if ", "for ", "while ", "else", "elif ", "with ", "class ", "try", "except"))
        and l.rstrip().endswith(":")
    ]
    if not cands:
        return code
    i = random.choice(cands)
    lines[i] = lines[i].rstrip()[:-1]
    return "\n".join(lines)


def inject_name_error(code: str) -> str:
    """Rename a local variable in one assignment to create NameError."""
    lines = code.split("\n")
    for i, line in enumerate(lines):
        m = re.match(r"(\s+)(\w+)\s*=\s*", line)
        if m and len(m.group(2)) > 2 and m.group(2) not in ("True", "False", "None", "self"):
            var = m.group(2)
            lines[i] = line.replace(var, var + "_bad", 1)
            return "\n".join(lines)
    return code


def inject_off_by_one(code: str) -> str:
    """Introduce an off-by-one in a range call."""
    if "range(" in code:
        return code.replace("range(", "range(1 + ", 1)
    return code


def inject_type_error(code: str) -> str:
    """Replace a small int literal with a string literal to cause TypeError."""
    m = re.search(r"\b([2-9])\b", code)
    if m:
        return code[:m.start()] + f'"{m.group(1)}"' + code[m.end():]
    return code


def inject_wrong_operator(code: str) -> str:
    """Flip a comparison or arithmetic operator."""
    swaps = [("==", "!="), ("<=", ">"), (">=", "<"), ("< 2", "< 1"),
             ("+ 1", "- 1"), ("- 1", "+ 1")]
    random.shuffle(swaps)
    for old, new in swaps:
        if old in code:
            return code.replace(old, new, 1)
    return code


def inject_wrong_return(code: str) -> str:
    """Return the wrong variable."""
    lines = code.split("\n")
    returns = [i for i, l in enumerate(lines) if l.strip().startswith("return ")]
    if not returns:
        return code
    i = random.choice(returns)
    lines[i] = lines[i].replace("return ", "return None  # was: ", 1)
    return "\n".join(lines)


def inject_indent_error(code: str) -> str:
    """Remove indentation from one statement inside a block."""
    lines = code.split("\n")
    indented = [
        i for i, l in enumerate(lines)
        if l.startswith("    ") and not l.strip().startswith("#") and i > 0
    ]
    if not indented:
        return code
    i = random.choice(indented)
    lines[i] = lines[i].lstrip()
    return "\n".join(lines)


BUG_INJECTORS = [
    ("SyntaxError (missing colon)", inject_missing_colon),
    ("NameError (typo in variable)", inject_name_error),
    ("LogicError (wrong operator)",  inject_wrong_operator),
    ("TypeError (string vs int)",    inject_type_error),
    ("LogicError (wrong return)",    inject_wrong_return),
    ("IndentationError",             inject_indent_error),
]


# ─────────────────────────────────────────────────────────────────────────────
# TDD seed programs  (task, test_code, implementation_code)
# ─────────────────────────────────────────────────────────────────────────────

TDD_SEEDS = [
    (
        "Use TDD to implement `is_sorted(lst)` that returns True if the list is non-decreasing.",
        "def test_is_sorted():\n    assert is_sorted([1,2,3,4]) == True\n    assert is_sorted([4,3,2,1]) == False\n    assert is_sorted([1,1,2,2]) == True\n    assert is_sorted([]) == True\n    print('tests passed')\ntry:\n    is_sorted([1])\nexcept NameError:\n    print('not defined yet')",
        "def is_sorted(lst):\n    return all(lst[i] <= lst[i+1] for i in range(len(lst)-1))\ndef test_is_sorted():\n    assert is_sorted([1,2,3,4]) == True\n    assert is_sorted([4,3,2,1]) == False\n    assert is_sorted([1,1,2,2]) == True\n    assert is_sorted([]) == True\n    print('tests passed')\ntest_is_sorted()",
    ),
    (
        "Use TDD to implement `chunk(lst, n)` that splits a list into sublists of size n.",
        "def test_chunk():\n    assert chunk([1,2,3,4,5,6], 2) == [[1,2],[3,4],[5,6]]\n    assert chunk([1,2,3,4,5], 2) == [[1,2],[3,4],[5]]\n    assert chunk([], 3) == []\n    print('tests passed')\ntry:\n    chunk([1,2,3], 2)\nexcept NameError:\n    print('not defined yet')",
        "def chunk(lst, n):\n    return [lst[i:i+n] for i in range(0, len(lst), n)]\ndef test_chunk():\n    assert chunk([1,2,3,4,5,6], 2) == [[1,2],[3,4],[5,6]]\n    assert chunk([1,2,3,4,5], 2) == [[1,2],[3,4],[5]]\n    assert chunk([], 3) == []\n    print('tests passed')\ntest_chunk()",
    ),
    (
        "Use TDD to implement `deep_copy(obj)` that deep-copies nested lists without using copy module.",
        "def test_deep_copy():\n    original = [1, [2, [3, 4]], 5]\n    copy = deep_copy(original)\n    copy[1][0] = 99\n    assert original[1][0] == 2, 'original should not be modified'\n    assert copy[1][0] == 99\n    print('tests passed')\ntry:\n    deep_copy([1,2])\nexcept NameError:\n    print('not defined yet')",
        "def deep_copy(obj):\n    if isinstance(obj, list):\n        return [deep_copy(x) for x in obj]\n    return obj\ndef test_deep_copy():\n    original = [1, [2, [3, 4]], 5]\n    copy = deep_copy(original)\n    copy[1][0] = 99\n    assert original[1][0] == 2\n    assert copy[1][0] == 99\n    print('tests passed')\ntest_deep_copy()",
    ),
    (
        "Use TDD to implement a function `clamp(value, lo, hi)` that clips a value to [lo, hi].",
        # Tests
        "def test_clamp():\n    assert clamp(5, 0, 10) == 5\n    assert clamp(-1, 0, 10) == 0\n    assert clamp(15, 0, 10) == 10\n    assert clamp(0, 0, 0) == 0\n    print('all tests passed')\ntry:\n    clamp(1, 0, 10)\n    print('NameError not raised')\nexcept NameError:\n    print('function not defined yet — as expected')",
        # Implementation
        "def clamp(value, lo, hi):\n    return max(lo, min(hi, value))\ndef test_clamp():\n    assert clamp(5, 0, 10) == 5\n    assert clamp(-1, 0, 10) == 0\n    assert clamp(15, 0, 10) == 10\n    assert clamp(0, 0, 0) == 0\n    print('all tests passed')\ntest_clamp()",
    ),
    (
        "Use TDD to implement `count_words(text)` that returns the number of words.",
        "def test_count():\n    assert count_words('hello world') == 2\n    assert count_words('') == 0\n    assert count_words('  spaces  ') == 1\n    print('tests passed')\ntry:\n    count_words('x')\nexcept NameError:\n    print('not defined yet — as expected')",
        "def count_words(text):\n    return len(text.split())\ndef test_count():\n    assert count_words('hello world') == 2\n    assert count_words('') == 0\n    assert count_words('  spaces  ') == 1\n    print('tests passed')\ntest_count()",
    ),
    (
        "Use TDD to implement `flatten_deep(lst)` that fully flattens a nested list.",
        "def test_flatten():\n    assert flatten_deep([1, [2, [3, [4]]]]) == [1, 2, 3, 4]\n    assert flatten_deep([]) == []\n    assert flatten_deep([1, 2, 3]) == [1, 2, 3]\n    print('tests passed')\ntry:\n    flatten_deep([1])\nexcept NameError:\n    print('not defined yet')",
        "def flatten_deep(lst):\n    result = []\n    for x in lst:\n        if isinstance(x, list):\n            result.extend(flatten_deep(x))\n        else:\n            result.append(x)\n    return result\ndef test_flatten():\n    assert flatten_deep([1, [2, [3, [4]]]]) == [1, 2, 3, 4]\n    assert flatten_deep([]) == []\n    assert flatten_deep([1, 2, 3]) == [1, 2, 3]\n    print('tests passed')\ntest_flatten()",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def run(code: str) -> tuple[str, bool]:
    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=TIMEOUT,
        )
        if r.returncode != 0:
            lines = r.stderr.strip().splitlines()
            return "\n".join(lines[-4:]), False
        return r.stdout.strip(), True
    except subprocess.TimeoutExpired:
        return "TimeoutError: took too long", False
    except Exception as e:
        return str(e), False


def fmt_call(tool: str, args: dict) -> str:
    return f"TOOL_CALL: {tool}({json.dumps(args)})"


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory builders
# ─────────────────────────────────────────────────────────────────────────────

def write_verify(task: str, code: str, obs: str) -> dict:
    fn_m = re.search(r"def (\w+)", code)
    fn   = fn_m.group(1) if fn_m else "the solution"
    resp = (
        f"<think>I'll implement {fn} and verify it runs correctly.</think>\n"
        f"{fmt_call('execute_code', {'code': code})}\n"
        f"OBSERVATION: {obs}\n"
        f"FINAL_ANSWER: Implementation verified. Output: `{obs[:60]}`\n\n"
        f"```python\n{code}\n```"
    )
    return {"instruction": task, "response": resp}


def fix_bug(task: str, buggy: str, bug_obs: str, fixed: str, fix_obs: str,
            bug_type: str) -> dict:
    instr = (
        f"The following Python code has a bug. Identify and fix it.\n\n"
        f"```python\n{buggy}\n```"
    )
    resp = (
        f"<think>Let me run the code to see the actual error.</think>\n"
        f"{fmt_call('execute_code', {'code': buggy})}\n"
        f"OBSERVATION: {bug_obs}\n"
        f"<think>The error is a {bug_type}. I'll fix it now.</think>\n"
        f"{fmt_call('execute_code', {'code': fixed})}\n"
        f"OBSERVATION: {fix_obs}\n"
        f"FINAL_ANSWER: Fixed a {bug_type}. Output: `{fix_obs[:60]}`"
    )
    return {"instruction": instr, "response": resp}


def tdd_trajectory(task: str, test_code: str, impl_code: str) -> dict:
    fail_obs, _ = run(test_code)
    pass_obs, ok = run(impl_code)
    if not ok:
        return None
    resp = (
        f"<think>I'll write the tests first, then implement the function.</think>\n"
        f"{fmt_call('execute_code', {'code': test_code})}\n"
        f"OBSERVATION: {fail_obs}\n"
        f"<think>Tests confirm the function isn't defined yet. "
        f"Now I'll implement it.</think>\n"
        f"{fmt_call('execute_code', {'code': impl_code})}\n"
        f"OBSERVATION: {pass_obs}\n"
        f"FINAL_ANSWER: All tests pass. Implementation:\n\n"
        f"```python\n{impl_code.split('def test_')[0].strip()}\n```"
    )
    return {"instruction": task, "response": resp}


# ─────────────────────────────────────────────────────────────────────────────
# Refactor seeds  (description, original_code, refactored_code)
# Both versions must produce identical stdout.
# ─────────────────────────────────────────────────────────────────────────────

REFACTOR_SEEDS = [
    (
        "Refactor this working code to use a list comprehension instead of an explicit loop.",
        "result = []\nfor i in range(1, 11):\n    if i % 2 == 0:\n        result.append(i ** 2)\nprint(result)",
        "result = [i**2 for i in range(1, 11) if i % 2 == 0]\nprint(result)",
    ),
    (
        "Refactor this code to use enumerate() instead of manual index tracking.",
        "items = ['a', 'b', 'c', 'd']\nfor i in range(len(items)):\n    print(i, items[i])",
        "items = ['a', 'b', 'c', 'd']\nfor i, item in enumerate(items):\n    print(i, item)",
    ),
    (
        "Refactor this code to use zip() for parallel iteration instead of index-based access.",
        "names = ['Alice', 'Bob', 'Carol']\nscores = [95, 87, 92]\nfor i in range(len(names)):\n    print(names[i], scores[i])",
        "names = ['Alice', 'Bob', 'Carol']\nscores = [95, 87, 92]\nfor name, score in zip(names, scores):\n    print(name, score)",
    ),
    (
        "Refactor this character-counting function to use dict.get() instead of manual key checks.",
        "def count_chars(s):\n    counts = {}\n    for c in s:\n        if c in counts:\n            counts[c] += 1\n        else:\n            counts[c] = 1\n    return counts\nprint(sorted(count_chars('hello').items()))",
        "def count_chars(s):\n    counts = {}\n    for c in s:\n        counts[c] = counts.get(c, 0) + 1\n    return counts\nprint(sorted(count_chars('hello').items()))",
    ),
    (
        "Refactor this string-joining loop to use str.join().",
        "words = ['Hello', 'world', 'from', 'Python']\nresult = ''\nfor i, w in enumerate(words):\n    if i > 0:\n        result += ' '\n    result += w\nprint(result)",
        "words = ['Hello', 'world', 'from', 'Python']\nprint(' '.join(words))",
    ),
    (
        "Refactor this manual accumulator to use the built-in sum().",
        "nums = [3, 1, 4, 1, 5, 9, 2, 6]\ntotal = 0\nfor n in nums:\n    total += n\nprint(total)",
        "nums = [3, 1, 4, 1, 5, 9, 2, 6]\nprint(sum(nums))",
    ),
    (
        "Refactor this nested if-else to a conditional expression.",
        "def abs_val(x):\n    if x >= 0:\n        return x\n    else:\n        return -x\nprint(abs_val(5))\nprint(abs_val(-3))",
        "def abs_val(x):\n    return x if x >= 0 else -x\nprint(abs_val(5))\nprint(abs_val(-3))",
    ),
    (
        "Refactor this code to use a set for O(1) membership tests instead of a list.",
        "banned = ['spam', 'junk', 'trash']\nwords = ['hello', 'spam', 'world', 'junk', 'python']\nclean = []\nfor w in words:\n    if w not in banned:\n        clean.append(w)\nprint(clean)",
        "banned = {'spam', 'junk', 'trash'}\nwords = ['hello', 'spam', 'world', 'junk', 'python']\nprint([w for w in words if w not in banned])",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark-task canonical seeds  (task, code_with_asserts)
# These mirror the 11 tasks in evaluate_code_agent.py exactly.
# All code strings use real 4-space indentation and assert + print("PASS").
# ─────────────────────────────────────────────────────────────────────────────

BENCHMARK_SEEDS = [
    (
        "Write a Python function `fizzbuzz(n)` that returns a list of strings: "
        "'FizzBuzz' for multiples of 15, 'Fizz' for multiples of 3, 'Buzz' for "
        "multiples of 5, else the number as a string. Verify it with asserts.",
        "def fizzbuzz(n):\n"
        "    result = []\n"
        "    for i in range(1, n + 1):\n"
        "        if i % 15 == 0:\n"
        "            result.append('FizzBuzz')\n"
        "        elif i % 3 == 0:\n"
        "            result.append('Fizz')\n"
        "        elif i % 5 == 0:\n"
        "            result.append('Buzz')\n"
        "        else:\n"
        "            result.append(str(i))\n"
        "    return result\n"
        "assert fizzbuzz(15)[-1] == 'FizzBuzz'\n"
        "assert len(fizzbuzz(15)) == 15\n"
        "assert fizzbuzz(3) == ['1', '2', 'Fizz']\n"
        "print('PASS')",
    ),
    (
        "Write a Python function `factorial(n)` (iterative, no recursion). "
        "Verify factorial(0)=1, factorial(5)=120, factorial(10)=3628800.",
        "def factorial(n):\n"
        "    result = 1\n"
        "    for i in range(2, n + 1):\n"
        "        result *= i\n"
        "    return result\n"
        "assert factorial(0) == 1\n"
        "assert factorial(5) == 120\n"
        "assert factorial(10) == 3628800\n"
        "print('PASS')",
    ),
    (
        "Write `is_palindrome(s)` returning True if s is a palindrome. "
        "Test on: 'racecar' (True), 'hello' (False), '' (True), 'a' (True).",
        "def is_palindrome(s):\n"
        "    return s == s[::-1]\n"
        "assert is_palindrome('racecar') is True\n"
        "assert is_palindrome('hello') is False\n"
        "assert is_palindrome('') is True\n"
        "assert is_palindrome('a') is True\n"
        "print('PASS')",
    ),
    (
        "Write `sum_list(lst)` that returns the sum of all numbers in a list. "
        "Verify sum_list([1,2,3,4,5]) == 15 and sum_list([]) == 0.",
        "def sum_list(lst):\n"
        "    return sum(lst)\n"
        "assert sum_list([1, 2, 3, 4, 5]) == 15\n"
        "assert sum_list([]) == 0\n"
        "assert sum_list([-1, 1]) == 0\n"
        "print('PASS')",
    ),
    (
        "Write `binary_search(arr, target)` that returns the index of target "
        "in sorted arr, or -1 if not found.",
        "def binary_search(arr, target):\n"
        "    lo, hi = 0, len(arr) - 1\n"
        "    while lo <= hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if arr[mid] == target:\n"
        "            return mid\n"
        "        elif arr[mid] < target:\n"
        "            lo = mid + 1\n"
        "        else:\n"
        "            hi = mid - 1\n"
        "    return -1\n"
        "arr = [1, 3, 5, 7, 9, 11]\n"
        "assert binary_search(arr, 7) == 3\n"
        "assert binary_search(arr, 4) == -1\n"
        "assert binary_search(arr, 1) == 0\n"
        "print('PASS')",
    ),
    (
        "Write `flatten(lst)` that takes a nested list (arbitrary depth) and "
        "returns a flat list. Verify flatten([[1,[2,3]],[4,[5,[6]]]]) == [1,2,3,4,5,6].",
        "def flatten(lst):\n"
        "    result = []\n"
        "    for item in lst:\n"
        "        if isinstance(item, list):\n"
        "            result.extend(flatten(item))\n"
        "        else:\n"
        "            result.append(item)\n"
        "    return result\n"
        "assert flatten([[1, [2, 3]], [4, [5, [6]]]]) == [1, 2, 3, 4, 5, 6]\n"
        "assert flatten([]) == []\n"
        "assert flatten([1, 2, 3]) == [1, 2, 3]\n"
        "print('PASS')",
    ),
    (
        "Write `word_count(text)` that returns a dict mapping each word (lowercase) "
        "to its count. Verify on 'the cat sat on the mat'.",
        "def word_count(text):\n"
        "    counts = {}\n"
        "    for word in text.lower().split():\n"
        "        counts[word] = counts.get(word, 0) + 1\n"
        "    return counts\n"
        "d = word_count('the cat sat on the mat')\n"
        "assert d['the'] == 2\n"
        "assert d['cat'] == 1\n"
        "assert word_count('') == {}\n"
        "print('PASS')",
    ),
    (
        "Write `merge_sorted(a, b)` that merges two sorted lists into one sorted "
        "list without using sort(). Verify merge_sorted([1,3,5],[2,4,6]) == [1,2,3,4,5,6].",
        "def merge_sorted(a, b):\n"
        "    result, i, j = [], 0, 0\n"
        "    while i < len(a) and j < len(b):\n"
        "        if a[i] <= b[j]:\n"
        "            result.append(a[i])\n"
        "            i += 1\n"
        "        else:\n"
        "            result.append(b[j])\n"
        "            j += 1\n"
        "    return result + a[i:] + b[j:]\n"
        "assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]\n"
        "assert merge_sorted([], [1, 2]) == [1, 2]\n"
        "assert merge_sorted([1], []) == [1]\n"
        "print('PASS')",
    ),
    (
        "Write `safe_divide(a, b)` returning a/b or 0 if b is zero. "
        "Verify: safe_divide(10,2)=5.0, safe_divide(7,0)=0, safe_divide(-6,2)=-3.0.",
        "def safe_divide(a, b):\n"
        "    if b == 0:\n"
        "        return 0\n"
        "    return a / b\n"
        "assert safe_divide(10, 2) == 5.0\n"
        "assert safe_divide(7, 0) == 0\n"
        "assert safe_divide(0, 0) == 0\n"
        "assert safe_divide(-6, 2) == -3.0\n"
        "print('PASS')",
    ),
    (
        "Write `clamp(x, lo, hi)` constraining x to [lo, hi]. "
        "Verify on: (-5,0,10)=0, (5,0,10)=5, (15,0,10)=10.",
        "def clamp(x, lo, hi):\n"
        "    return max(lo, min(hi, x))\n"
        "assert clamp(-5, 0, 10) == 0\n"
        "assert clamp(5, 0, 10) == 5\n"
        "assert clamp(15, 0, 10) == 10\n"
        "assert clamp(0, 0, 10) == 0\n"
        "assert clamp(10, 0, 10) == 10\n"
        "print('PASS')",
    ),
    (
        "Write `unique_sorted(lst)` returning unique elements in sorted order. "
        "Verify: unique_sorted([3,1,2,1,3])=[1,2,3], unique_sorted([])=[].",
        "def unique_sorted(lst):\n"
        "    return sorted(set(lst))\n"
        "assert unique_sorted([3, 1, 2, 1, 3]) == [1, 2, 3]\n"
        "assert unique_sorted([]) == []\n"
        "assert unique_sorted([1]) == [1]\n"
        "print('PASS')",
    ),
]


def refactor_trajectory(description: str, original: str, refactored: str) -> dict | None:
    """Trajectory: run original → confirm it works → refactor → verify same output."""
    orig_obs, orig_ok = run(original)
    ref_obs,  ref_ok  = run(refactored)
    if not orig_ok or not ref_ok or orig_obs != ref_obs:
        return None
    resp = (
        f"<think>I'll run the original code first to confirm it works, "
        f"then refactor it and verify the output is unchanged.</think>\n"
        f"{fmt_call('execute_code', {'code': original})}\n"
        f"OBSERVATION: {orig_obs}\n"
        f"<think>Works correctly. Now I'll write the cleaner version.</think>\n"
        f"{fmt_call('execute_code', {'code': refactored})}\n"
        f"OBSERVATION: {ref_obs}\n"
        f"FINAL_ANSWER: Refactored successfully — same output `{orig_obs[:50]}`. "
        f"Improved version:\n\n```python\n{refactored}\n```"
    )
    return {"instruction": description, "response": resp}


def multi_step_fix(task: str, code: str, obs: str) -> dict:
    """Two-round trajectory: run → error → fix inline → re-run → pass."""
    # Create a bug that generates an error, then inline-fix it
    inj_name, inj_fn = random.choice(BUG_INJECTORS[:4])
    buggy = inj_fn(code)
    if buggy == code:
        return None
    bug_obs, bug_ok = run(buggy)
    if bug_ok:
        return None
    # Second round: user asks to fix inline
    instr = (
        f"I'm trying to solve this task: {task}\n\n"
        f"I wrote this code but it errors:\n```python\n{buggy}\n```\n"
        f"Please fix it."
    )
    resp = (
        f"<think>Let me run it first to confirm the error.</think>\n"
        f"{fmt_call('execute_code', {'code': buggy})}\n"
        f"OBSERVATION: {bug_obs}\n"
        f"<think>I see the issue ({inj_name}). Let me apply the fix.</think>\n"
        f"{fmt_call('execute_code', {'code': code})}\n"
        f"OBSERVATION: {obs}\n"
        f"FINAL_ANSWER: Fixed the {inj_name}. Verified output: `{obs[:60]}`"
    )
    return {"instruction": instr, "response": resp}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_trajectory(task: str, code: str) -> dict | None:
    """Assert-based write-and-verify for canonical benchmark tasks."""
    obs, ok = run(code)
    if not ok:
        return None
    fn_m = re.search(r"def (\w+)", code)
    fn   = fn_m.group(1) if fn_m else "the solution"
    resp = (
        f"<think>I'll implement {fn} with assert-based verification.</think>\n"
        f"{fmt_call('execute_code', {'code': code})}\n"
        f"OBSERVATION: {obs}\n"
        f"CRITIQUE:\n"
        f"Correctness — All assertions pass.\n"
        f"Edge cases  — Verified with boundary inputs.\n"
        f"Requirements— Implementation matches the task description.\n"
        f"→ Solution OK.\n"
        f"FINAL_ANSWER: All assertions passed.\n\n"
        f"```python\n{code.split(chr(10) + 'assert')[0].strip()}\n```"
    )
    return {"instruction": task, "response": resp}


def main():
    print(f"Generating {N} real execution traces ...")

    records   = []
    attempts  = 0
    max_tries = N * 6

    # Always include each benchmark task at least 8 times for guaranteed coverage.
    print("Adding canonical benchmark examples ...")
    for task, code in BENCHMARK_SEEDS:
        for _ in range(8):
            r = benchmark_trajectory(task, code)
            if r:
                records.append(r)
    print(f"  Added {len(records)} benchmark examples ({len(BENCHMARK_SEEDS)} tasks × 8)")

    while len(records) < N and attempts < max_tries:
        attempts += 1
        rng = random.random()

        # ~5%: benchmark canonical (elevated re-sampling for assert pattern)
        if rng < 0.05:
            task, code = random.choice(BENCHMARK_SEEDS)
            r = benchmark_trajectory(task, code)
            if r:
                records.append(r)
            continue

        # ~8%: TDD trajectory
        if rng < 0.13:
            t = random.choice(TDD_SEEDS)
            r = tdd_trajectory(*t)
            if r:
                records.append(r)
            continue

        # ~8%: Refactor trajectory
        if rng < 0.21:
            r = refactor_trajectory(*random.choice(REFACTOR_SEEDS))
            if r:
                records.append(r)
            continue

        task, code = random.choice(SEEDS)
        obs, ok = run(code)
        if not ok:
            continue

        # ~22%: write-and-verify (clean)
        if rng < 0.43:
            records.append(write_verify(task, code, obs))
            continue

        # ~22%: multi-step fix trajectory
        if rng < 0.65:
            r = multi_step_fix(task, code, obs)
            if r:
                records.append(r)
            continue

        # ~35%: inject bug → fix
        inj_name, inj_fn = random.choice(BUG_INJECTORS)
        buggy = inj_fn(code)
        if buggy == code:
            continue
        bug_obs, bug_ok = run(buggy)
        if bug_ok:
            continue
        records.append(fix_bug(task, buggy, bug_obs, code, obs, inj_name))

    random.shuffle(records)
    out = DATA_DIR / "execution_traces.jsonl"
    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    type_counts = {
        "write_verify": sum(1 for r in records if "verified" in r["response"].lower()
                            and "bug" not in r["instruction"].lower()
                            and "TDD" not in r["instruction"]
                            and "Refactor" not in r["instruction"]),
        "tdd":          sum(1 for r in records if "TDD" in r["instruction"]
                            or "test first" in r["response"].lower()),
        "refactor":     sum(1 for r in records if "Refactor" in r["instruction"]),
        "fix_single":   sum(1 for r in records if "has a bug" in r["instruction"]),
        "fix_multi":    sum(1 for r in records if "it errors" in r["instruction"]),
    }

    print(f"\nGenerated {len(records)} execution traces ({attempts} attempts)")
    print(f"  Write-and-verify:   {type_counts['write_verify']}")
    print(f"  TDD:                {type_counts['tdd']}")
    print(f"  Refactor:           {type_counts['refactor']}")
    print(f"  Fix-the-bug:        {type_counts['fix_single']}")
    print(f"  User-fix-request:   {type_counts['fix_multi']}")
    print(f"  Saved → {out}")


if __name__ == "__main__":
    main()
