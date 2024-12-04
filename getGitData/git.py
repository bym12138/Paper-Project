# import requests
# import json

# # GitHub API URL for repositories sorted by stars
# url = "https://api.github.com/search/repositories?q=stars:%3E300&sort=stars"

# # Make a GET request to the GitHub API
# response = requests.get(url)

# # Convert the response to JSON
# data = response.json()
# i = 0 
# # Open the output file
# with open('/data/home2/kjb/bym/project/BART/getGitData/output.jsonl', 'w') as f:
#     # For each item in the data, print the necessary information
#     for item in data['items']:
#         # Get the releases for the repository
#         releases_url = item['releases_url'].replace('{/id}', '')
#         releases_response = requests.get(releases_url)
#         releases_data = releases_response.json()

#         # Check if the repository has releases
#         if releases_data:
#             # For each release, print the necessary information
#             for release in releases_data:
#                 # Write the data to the file
#                 f.write(json.dumps({
#                     "repo": item['full_name'],
#                     "stars": item['stargazers_count'],
#                     "language": item['language'],
#                     "note_url": release['html_url'],
#                     "commits_url": item['compare_url']
#                 }) + '\n')
#             i += 1   
#             if i > 20:
#                 break
            
import requests
import json

# Your GitHub Personal Access Token
token = ''

# GitHub API URL for repositories sorted by stars
url = "https://api.github.com/search/repositories?q=stars:%3E300&sort=stars"

# Headers for the API request
headers = {'Authorization': f'token {token}'}

# Make a GET request to the GitHub API
response = requests.get(url, headers=headers)

# Convert the response to JSON
data = response.json()
 
# Open the output file
with open('C:/Users/Administrator/Documents/Mr.b/python本地/project/BART/getGitData/rnsum2.jsonl', 'w') as f:
    # For each item in the data, print the necessary information
    for item in data['items']:
        # Get the releases for the repository
        releases_url = item['releases_url'].replace('{/id}', '')
        releases_response = requests.get(releases_url, headers=headers)
        releases_data = releases_response.json()

        # Check if the repository has releases
        if releases_data:
            # Get the tags for the repository
            tags_url = item['tags_url']
            tags_response = requests.get(tags_url, headers=headers)
            tags_data = tags_response.json()

            # For each release, print the necessary information
            for i, release in enumerate(releases_data):
                # Get the commit sha for this release and the previous one
                if i < len(tags_data) - 1:
                    base = tags_data[i+1]['commit']['sha']
                    head = tags_data[i]['commit']['sha']

                    # Write the data to the file
                    f.write(json.dumps({
                        "repo": item['full_name'],
                        "stars": item['stargazers_count'],
                        "language": item['language'],
                        "note_url": release['html_url'],
                        "commits_url": f"https://github.com/{item['full_name']}/compare/{base}...{head}"
                    }) + '\n')
            if i > 20:
                break
