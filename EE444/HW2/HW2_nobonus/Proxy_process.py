import socket
import time
from collections import deque

# Constants
HOST = '127.0.0.1'
CLIENT_PORT = 6001
SERVER_PORT = 6002
CACHE_SIZE = 5

# Create sockets
proxy_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
proxy_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Configure address reuse and bind the sockets
proxy_client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
proxy_client_socket.bind((HOST, CLIENT_PORT))
proxy_server_socket.connect((HOST, SERVER_PORT))

# Listen for client connections
proxy_client_socket.listen(1)
print("Proxy listening for connections on port", CLIENT_PORT)
conn, client_address = proxy_client_socket.accept()
print('Connected by', client_address)

# Initialize the cache
cache = deque(maxlen=CACHE_SIZE)

# Helper function to send messages and receive responses from the server
def send_message_to_server(message):
    proxy_server_socket.sendall(bytes(message, 'utf-8'))
    time.sleep(0.5)
    data_received = proxy_server_socket.recv(1024)
    data_received = data_received.decode('utf-8')
    return data_received

while True:
    # Receive messages from the client
    message = conn.recv(1024).decode('utf-8')
    print(f"Received message from client: {message}")

    # Parse the message
    fields = message.split(';')
    op = fields[0].split('=')[1]
    if op != "CLR":
        indices = list(map(int, fields[1].split('=')[1].split(',')))

    if op == "GET" or op == "ADD":
        # Check if all indices are in cache
        all_indices_in_cache = all(index in (entry[0] for entry in cache) for index in indices)

        if not all_indices_in_cache:
            # Request missing indices from the server
            missing_indices = [index for index in indices if index not in (entry[0] for entry in cache)]
            missing_indices_str = ','.join(map(str, missing_indices))
            request_message = f"OP=GET;IND={missing_indices_str};"
            response = send_message_to_server(request_message)
            print(f"Received response from server: {response}")

            # Update the cache with the received data
            data_received = response.split(';')[1].split('=')[1].split(',')
            for index, data in zip(missing_indices, data_received):
                cache.append((index, int(data)))

        # Handle the operation using the updated cache
        if op == "GET":
            response_data = {}
            for index in indices:
                for entry in cache:
                    if entry[0] == index:
                        response_data[index] = entry[1]
                        break
            data = [response_data[index] for index in indices]
            # data = [entry[1] for entry in cache if entry[0] in indices]
            response = f"OP=GET;DATA={','.join(map(str, data))};"
        elif op == "ADD":
            data = sum(entry[1] for entry in cache if entry[0] in indices)
            response = f"OP=ADD;DATA={data};"
    elif op == "PUT":
        # Update the cache with the new data
        data = list(map(int, fields[2].split('=')[1].split(',')))
        for index, new_data in zip(indices, data):
            if index in (entry[0] for entry in cache):
                cache[cache.index((index, next(entry[1] for entry in cache if entry[0] == index)))]= (index, new_data)

        # Forward the update request to the server
        response = send_message_to_server(message)
        print(f"Received response from server: {response}")
    elif op == "CLR":
        # Clear the cache and forward the clear request to the server
        cache.clear()
        response = send_message_to_server(message)
        print(f"Received response from server: {response}")
    else:
        response = "Invalid operation"

    # Send the response back to the client
    print(f"Sending response to client: {response}")
    conn.sendall(bytes(response, 'utf-8'))

    # Print the current cache state
    print("Current cache state:")
    print(cache)
