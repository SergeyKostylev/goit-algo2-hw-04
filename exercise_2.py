from trie import Trie

class Homework(Trie):
    def count_words_with_suffix(self, pattern) -> int:
        if not isinstance(pattern, str):
            raise TypeError("pattern має бути рядком")

        def dfs(node, suffix):
            nonlocal count
            # Якщо це кінець слова, перевіряємо суфікс
            if node.value is not None:
                if suffix.endswith(pattern):
                    count += 1
            # Рекурсивно проходимо по всіх дітях
            for char, child in node.children.items():
                dfs(child, suffix + char)

        # Обробка випадку порожнього суфікса
        if pattern == "":
            count = 0
            def count_all(node):
                nonlocal count
                if node.value is not None:
                    count += 1
                for child in node.children.values():
                    count_all(child)
            count_all(self.root)
            return count

        count = 0
        dfs(self.root, "")
        return count

    def has_prefix(self, prefix) -> bool:
        if not isinstance(prefix, str):
            raise TypeError("prefix має бути рядком")

        def has_word(node):
            if node.value is not None:
                return True
            for child in node.children.values():
                if has_word(child):
                    return True
            return False

        # Обробка порожнього префікса
        if prefix == "":
            return has_word(self.root)

        # Знаходимо вузол, що відповідає префіксу
        current = self.root
        for char in prefix:
            if char not in current.children:
                return False
            current = current.children[char]

        # Перевіряємо, чи є слова після префікса
        return has_word(current)

if __name__ == "__main__":
    trie = Homework()
    words = ["apple", "application", "banana", "cat"]
    for i, word in enumerate(words):
        trie.put(word, i)

    # Перевірка кількості слів, що закінчуються на заданий суфікс
    assert trie.count_words_with_suffix("e") == 1  # apple
    assert trie.count_words_with_suffix("ion") == 1  # application
    assert trie.count_words_with_suffix("a") == 1  # banana
    assert trie.count_words_with_suffix("at") == 1  # cat

    # Перевірка наявності префікса
    assert trie.has_prefix("app") == True  # apple, application
    assert trie.has_prefix("bat") == False
    assert trie.has_prefix("ban") == True  # banana
    assert trie.has_prefix("ca") == True  # cat