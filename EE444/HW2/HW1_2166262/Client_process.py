import socket
import time

# Constants
HOST = '127.0.0.1'
PORT = 6001

# Create a socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))

# Helper function to send messages and receive responses
def send_message(message):
    client_socket.sendall(bytes(message, 'utf-8'))
    time.sleep(0.5)
    data_received = client_socket.recv(1024)
    data_received = data_received.decode('utf-8')
    return data_received

while True:
    operation = input("Enter operation (GET, PUT, CLR, ADD, DEL, SRH, FLS): ").upper()

    if operation == "GET" or operation == "PUT" or operation == "ADD" or operation == "DEL":
        indices = input("Enter indices separated by commas (e.g., 1,2,3): ")
        if operation == "PUT":
            data = input("Enter data to update separated by commas (e.g., 100,200,300): ")
            message = f"OP={operation};IND={indices};DATA={data};"
        else:
            message = f"OP={operation};IND={indices};"
    elif operation == "CLR":
        message = f"OP={operation};"
    elif operation == "SRH":
        data = input("Enter data to search (e.g., 100): ")
        message = f"OP={operation};DATA={data};"
    elif operation == "FLS":
        message = f"OP={operation};"
    else:
        print("Invalid operation. Please enter GET, PUT, CLR, ADD, DEL, SRH, or FLS.")
        continue

    print(f"Sending message: {message}")
    response = send_message(message)
    print(f"Received response: {response}")
    print("#############################################")
