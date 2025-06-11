import torch
import time

# This is your stream (queue of events)
event_stream = []

# Simulate adding events to the stream
def add_event(event_name):
    print(f"📩 Event created: {event_name}")
    event_stream.append(event_name)

# Handle the events one by one (FIFO)
def handle_events():
    while event_stream:
        current_event = event_stream.pop(0)
        print(f"⚙️  Handling event: {current_event}")
        time.sleep(1)  # simulate time taken to handle
        print(f"✅ Finished: {current_event}\n")

# Main part
add_event("User clicked Button")
add_event("New message received")
add_event("File uploaded")

handle_events()

g_cpu = torch.Generator()
g_cuda = torch.Generator(device='cpu')

print(g_cpu,g_cuda)

g_cuda = torch.Generator(device='cpu')
cloned_state = g_cuda.clone_state()

print(g_cpu,cloned_state)

print("\n🎲 Set Seeds")
torch.manual_seed(42)                    # Sets seed for all devices
torch.cuda.manual_seed_all(42)          # Sets seed for all CUDA devices (if using GPU)
torch.seed()                            # Sets seed to a non-deterministic value
print("Initial seed:", torch.initial_seed())

print("\n🎲 Get and Set RNG State")
rng_state = torch.get_rng_state()       # Save current RNG state
print("RNG State (first 5 bytes):", rng_state[:5])

# Modify state and restore it
torch.rand(1)
torch.set_rng_state(rng_state)
print("RNG restored:", torch.rand(1))  # Same as before restoring

print("\n🎲 Default Generator")
print("Default generator:", torch.default_generator)

print("\n🎲 Random Distributions")

# Bernoulli (probability of 0.7 for 1)
probs = torch.tensor([0.7, 0.3, 0.5])
print("Bernoulli:", torch.bernoulli(probs))

# Multinomial
weights = torch.tensor([[0.1, 0.3, 0.6]])
print("Multinomial (1 sample):", torch.multinomial(weights, 1))
print("Multinomial (3 samples):", torch.multinomial(weights, 3))

# Normal Distribution
mean = torch.tensor([0.0, 1.0])
std = torch.tensor([1.0, 2.0])
print("Normal:", torch.normal(mean, std))

# Poisson
rate = torch.tensor([0.5, 1.5, 2.5])
print("Poisson:", torch.poisson(rate))

# Uniform [0, 1)
print("rand (2x3):", torch.rand(2, 3))
print("rand_like:", torch.rand_like(torch.zeros(2, 3)))

# Random Integers
print("randint (5-10):", torch.randint(5, 10, (2, 3)))
print("randint_like:", torch.randint_like(torch.zeros(2, 3, dtype=torch.int), 1, 4))

# Normal Distribution (mean=0, std=1)
print("randn (2x2):", torch.randn(2, 2))
print("randn_like:", torch.randn_like(torch.zeros(2, 2)))

# Random Permutation
print("randperm (5):", torch.randperm(5))
