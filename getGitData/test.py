import json
with open('/data/home2/kjb/bym/project/BART/getGitData/rnsum_test.jsonl', 'a') as f:
    for i in range(10 - 1):
        #if not isinstance(releases[i], dict):# or 'html_url' not in releases[i]:
            if i > 10:
                break
            print({
                'repo':111,
                'aaa':i
            })
            # Write the data to the file
            f.write(json.dumps({
            'repo':111,
             'aaa':i
            }) + '\n')