import requests

url = "http://127.0.0.1:8000/recommend-from-dataset"

# Simulate a user with strong reading engagement
payload = {
    "user_avg_rating": 4.2,
    "user_avg_pages_read": 320.0,
    "user_conversion_rate_90d": 0.40,
    "previous_interactions": 15,
    "previous_conversions_90d": 5,
    "top_n": 3  # Asking for top 3 recommendations
}

print("Sending request to the recommendation API...")
response = requests.post(url, json=payload)

print("Status Code:", response.status_code)
print("Raw Response Text:", response.text)  # This will show the exact error message from the server

# Try parsing JSON only if the request succeeded
if response.status_code == 200:
    print("\nTop Real Book Recommendations:")
    print(response.json())
else:
    print("\nRequest failed. Check the raw response text above for details.")