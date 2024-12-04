import requests
import json
import time
import os
from bs4 import BeautifulSoup
from requests.exceptions import SSLError

token = '  '

F_SYNONYMS = open('/data/home2/kjb/bym/project/BART/getGitData/words/features.txt').read().split('\n')
I_SYNONYMS = open('/data/home2/kjb/bym/project/BART/getGitData/words/improvements.txt').read().split('\n')
BF_SYNONYMS = open('/data/home2/kjb/bym/project/BART/getGitData/words/bug_fixes.txt').read().split('\n')
DR_SYNONYMS = open('/data/home2/kjb/bym/project/BART/getGitData/words/deprecations_removals.txt').read().split('\n')
ALL_SYNONYMS = F_SYNONYMS + I_SYNONYMS + BF_SYNONYMS + DR_SYNONYMS


def get_api_token():
    return token

def handle_rate_limit(response):
    if response.status_code == 429:
        time.sleep(int(response.headers['Retry-After']))
        return True
    return False

def get_repos():
    api_token = get_api_token()
    headers = {'Authorization': f'token {api_token}'}
    url = 'https://api.github.com/search/repositories?q=stars:>150&sort=stars&order=desc'
    while True:
        response = requests.get(url, headers=headers)
        if handle_rate_limit(response):
            continue
        return response.json()['items']

def get_releases(repo):
    api_token = get_api_token()
    headers = {'Authorization': f'token {api_token}'}
    url = f'https://api.github.com/repos/{repo}/releases'
    keywords = ALL_SYNONYMS
    try:
        response = requests.get(url, headers=headers)
    except Exception as e:
        print(f"get_releases_error {e}")
    if handle_rate_limit(response):
        print('速率限制')
    releases = response.json()
    if len(releases) < 2:
        # print('返回[]')
        return
    for release in releases:
        note_url = release['html_url']
        try:
            note_response = requests.get(note_url)
            soup = BeautifulSoup(note_response.text, 'html.parser')
            h3_tags = soup.find_all('h3')
            if any(keyword.lower() in tag.text.lower() for tag in h3_tags for keyword in keywords):
                print('1')
                yield release
            else:
                print('empty')
        except Exception as e:
            print(f"caocaocaocaocaocaocaocaocaocaocaocaocaocaocaocaocaocaocao: {e}")

def get_commits(repo, base, head):
    api_token = get_api_token()
    headers = {'Authorization': f'token {api_token}'}
    url = f'https://api.github.com/repos/{repo}/compare/{base}...{head}'
    while True:
        response = requests.get(url, headers=headers)
        if handle_rate_limit(response):
            continue
        return response.json()
    

def main():
    repos = get_repos()
    with open('/data/home2/kjb/bym/project/BART/getGitData/rnsum.jsonl', 'a') as f:
        for repo in repos:
            releases = list(get_releases(repo['full_name']))
            for i in range(len(releases) - 1):
                #if not isinstance(releases[i], dict):# or 'html_url' not in releases[i]:
                    if i > 10:
                        break
                    commits = get_commits(repo['full_name'], releases[i+1]['tag_name'], releases[i]['tag_name'])
                    print({
                        'repo': repo['full_name'],
                        'stars': repo['stargazers_count'],
                        'language': repo['language'],
                        'note_url': releases[i]['html_url'],
                        'commits_url': commits['html_url']
                    })
                    # Write the data to the file
                    f.write(json.dumps({
                    'repo': repo['full_name'],
                        'stars': repo['stargazers_count'],
                        'language': repo['language'],
                        'note_url': releases[i]['html_url'],
                        'commits_url': commits['html_url']
                    }) + '\n')
            

if __name__ == '__main__':
    main()