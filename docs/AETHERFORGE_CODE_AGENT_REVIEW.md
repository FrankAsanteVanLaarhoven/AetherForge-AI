
---

## 15. Real-tool parser fix and clean training audit

After the previous smoke tests showed invalid tool calls such as:

```text
TOOL_CALL: is_palindrome({"s": "racecar"})
TOOL_CALL: sum_list({"lst": [1,2,3,4,5]})
