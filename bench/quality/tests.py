"""Tests for the quality check in tasks.json: one function per task id, each raising on a failure.
The model never sees these; run.py asks the tasks and runs each answer against them."""


def t_intervals(ns):
    f = ns["merge_intervals"]
    assert f([]) == []
    assert f([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
    assert f([[1, 3], [3, 5]]) == [[1, 5]]
    assert f([[5, 6], [1, 2], [2, 4]]) == [[1, 4], [5, 6]]
    assert f([[1, 10], [2, 3], [4, 5]]) == [[1, 10]]
    assert [list(x) for x in f([[1, 4], [0, 0]])] == [[0, 0], [1, 4]]


def t_lru(ns):
    C = ns["LRUCache"]
    c = C(2)
    c.put(1, 1); c.put(2, 2)
    assert c.get(1) == 1
    c.put(3, 3)
    assert c.get(2) == -1
    c.put(4, 4)
    assert c.get(1) == -1
    assert c.get(3) == 3 and c.get(4) == 4
    c = C(2)
    c.put(1, 1); c.put(2, 2); c.put(1, 10)
    c.put(3, 3)
    assert c.get(2) == -1 and c.get(1) == 10
    c = C(1)
    c.put(1, 1); c.put(2, 2)
    assert c.get(1) == -1 and c.get(2) == 2


def t_duration(ns):
    f = ns["parse_duration"]
    assert f("2h") == 7200
    assert f("45m") == 2700
    assert f("90s") == 90
    assert f("1h30m") == 5400
    assert f("1h0m5s") == 3605
    assert f("10m5s") == 605
    for bad in ["", "1x", "h1", "1m1h", "1h1h", "abc"]:
        try:
            f(bad)
        except ValueError:
            continue
        raise AssertionError(f"no ValueError for {bad!r}")


def t_topk(ns):
    f = ns["top_k_frequent"]
    assert f(["i", "love", "leetcode", "i", "love", "coding"], 2) == ["i", "love"]
    w = ["the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"]
    assert f(w, 4) == ["the", "is", "sunny", "day"]
    assert f(["b", "a", "c"], 2) == ["a", "b"]


def t_rpn(ns):
    f = ns["eval_rpn"]
    assert f(["2", "1", "+", "3", "*"]) == 9
    assert f(["4", "13", "5", "/", "+"]) == 6
    assert f(["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]) == 22
    assert f(["7", "-2", "/"]) == -3
    assert f(["-7", "2", "/"]) == -3
    assert f(["3", "4", "-"]) == -1


def t_palindrome(ns):
    f = ns["longest_palindrome"]
    assert f("") == ""
    assert f("a") == "a"
    assert f("babad") == "bab"
    assert f("cbbd") == "bb"
    assert f("forgeeksskeegfor") == "geeksskeeg"
    assert f("abcd") == "a"


def t_wrap(ns):
    f = ns["wrap"]
    assert f("", 10) == []
    assert f("   ", 10) == []
    assert f("the quick brown fox", 10) == ["the quick", "brown fox"]
    assert f("a bb ccc", 4) == ["a bb", "ccc"]
    assert f("supercalifragilistic is long", 10) == ["supercalifragilistic", "is long"]
    assert f("one  two\nthree", 7) == ["one two", "three"]


def t_brackets(ns):
    f = ns["balanced"]
    assert f("") is True
    assert f("()[]{}") is True
    assert f("{[()]}") is True
    assert f("a(b[c]{d}e)f") is True
    assert f("(]") is False
    assert f("([)]") is False
    assert f("((") is False
    assert f(")(") is False
