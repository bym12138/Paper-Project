
categories = {
    "New Features": ["feat", "add", "added","adding", "new", "implement", "feature", "introduce", "create"],
    "Bug Fixes": ["fixes", "fix", "fixed", "bugfix", "resolve", "correct", "repair"],
    "Improvements": ["improve", "optimiz", "optimalization", "perf", "enhance","upgrade","performance"],
    "Update": ["bump","bumps","bumped","update"],
    "Security": ["security", "secure","unsecure", "vulnerability", "patch", "vulner"],
    "Changes": ["merge","change","changes", "refactor", "revert", "update", "modify", "adjust", "reconfigure", "alter"],
    "Deprecations and Removals": ["breaking", "deprecate", "remove","removed","removing", "discard", "obsolet", "discontinue", "no support", "no longer"],
    "Documentation and Tooling": ["readme", ".md", "ci", "changelog", "document", "documentation", "doc", "docs", "tool", "script update", "guide"],
    "Miscellaneous": ["chore", "other", "miscellaneous", "various", "test", "build","style",]
}

stats = {category: {keyword: 0 for keyword in keywords} for category, keywords in categories.items()}
category_totals = {category: 0 for category in categories}

with open('mult_class/data/rn.txt', 'r') as file:
    for line in file:
        release_note = line.lower() 

        for category, keywords in categories.items():
            for keyword in keywords:
                if keyword in release_note:
                    stats[category][keyword] += 1
                    category_totals[category] += 1

total_hits = sum(category_totals.values())
print("Keyword Hit Counts by Category:")
for category, keywords in stats.items():
    print(f"Category: {category}")
    for keyword, count in keywords.items():
        print(f"  Keyword '{keyword}' hit {count} times")
    category_count = category_totals[category]
    category_percentage = (category_count / total_hits) * 100 if total_hits > 0 else 0
    print(f"  Total hits in category '{category}': {category_count} ({category_percentage:.2f}%)\n")

print(f"Total hits across all categories: {total_hits}")
for category, count in category_totals.items():
    percentage = (count / total_hits) * 100 if total_hits > 0 else 0
    print(f"  Category '{category}' contributes {count} hits ({percentage:.2f}%)")