# Note: Since the name of the file starts with a number (due to the rule mentioned in ODTUCLASS), it might not work on some systems.
# In that case, please rename the file to something else.
import requests
import json
import random

def print_party_list():
    res = requests.get('http://127.0.0.1:5000/election/parties')
    parties = json.loads(res.text)
    print("Party List:")
    for party in parties:
        print(party['party_name'])

def add_party(party_name):
    data = {'party_name': party_name}
    res = requests.put('http://127.0.0.1:5000/election/parties', json=data)
    if("already registered" in res.text):
        msg = json.loads(res.text)
        print(f"{msg['message']} Status Code: {res.status_code}\n")
    else:
        print(f"{res.text}")

def delete_party(party_name):
    data = {'party_name': party_name}
    res = requests.delete('http://127.0.0.1:5000/election/parties', json=data)
    print(res.text)

def get_party_list():
    res = requests.get('http://127.0.0.1:5000/election/parties')
    parties = json.loads(res.text)
    return parties

def simulate_election(region, party_percentages):
    data = {'region': region, **party_percentages}
    res = requests.post('http://127.0.0.1:5000/election/simulate', json=data)
    seat_counts = json.loads(res.text)
    print("MP Distribution:")
    for party, seats in seat_counts.items():
        print(f"{party}: {seats} seats")
    print()

if __name__ == "__main__":
    # Pull the list of parties (should be empty)
    print("Initial Party List:")
    print_party_list()
    
    # Add parties
    input("\nPress Enter to continue... Next Step: Add Parties")
    add_party("Party1")
    add_party("Party2")
    add_party("Party3")
    add_party("Party4")

    # Print the updated party list
    print("Updated Party List:")
    print_party_list()

    # Try to re-add Party4 (should return an error)
    input("\nPress Enter to continue... Next Step: Add Party4")
    add_party("Party4")

    # Delete Party4
    input("\nPress Enter to continue... Next Step: Delete Party4")
    delete_party("Party4")

    # Print the remaining parties
    print("Updated Party List:")
    print_party_list()

    # Simulate the election for three different regions with vote percentages
    # Confirmed by https://icon.cat/util/elections (Set seats for region, add 3 candidatures and add votes)
    input("\nPress Enter to continue... Next Step: Simulate Election")
    regions_res = requests.get('http://127.0.0.1:5000/election/regions')
    regions = json.loads(regions_res.text)
    # Select three random regions
    regions = random.sample(regions,3)
    for i in range(3):
        # Select a region from the random 3 regions
        region = regions[i]['region_name']

        # Generate random vote percentages for the parties
        parties = get_party_list()

        # Generate random values for proportions
        proportion_values = []
        votes = 100
        for i in range(len(parties)-1):
            proportion = random.randint(0, votes)
            proportion_values.append(proportion)
            votes -= proportion
        proportion_values.append(votes) # Assign the remaining votes to the last party
        
        # Calculate the sum of proportion values
        total_proportion = sum(proportion_values)

        # Assign random proportions to the parties
        party_percentages = {}
        for party, proportion in zip(parties, proportion_values):
            percentage = round(proportion / total_proportion * 100, 2)
            party_percentages[party['party_name']] = percentage

        # Simulate the election and print the results
        print(f"Simulating election for {region} with the following vote percentages:")
        for party, percentage in party_percentages.items():
            print(f"{party}: {percentage}%")
        simulate_election(region, party_percentages)