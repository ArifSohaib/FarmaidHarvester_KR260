import zmq
import pickle

# Set up the ZMQ context and REQ (request) socket
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")  # Adjust the address as needed

try:
    while True:
        # Send a request to the server
        socket.send(b"Requesting boxes")
        
        # Receive the serialized boxes from the server
        message = socket.recv()

        # Deserialize the boxes using pickle
        boxes = pickle.loads(message)

        # Print the received boxes
        print(f"Received boxes: {boxes}")

        # Process the boxes as needed
        # ...

except KeyboardInterrupt:
    print("Client interrupted.")

finally:
    # Clean up the socket and context
    socket.close()
    context.term()
