import serial
import time
import re

# set serial port that connect to the headset
SERIAL_PORT = '/dev/cu.usbserial-10'
# set baud rate to 57600, as req.
BAUD_RATE = 57600
COLLECTION_DURATION = 60  

def get_user_confirmation(session_type, participant_id):
    print(f"\n{'='*60}")
    print(f"PARTICIPANT: {participant_id.upper()} | SESSION: {session_type.upper()} ({COLLECTION_DURATION} seconds)")
    print("="*60)
    
    if session_type == "relaxed":
        print("INSTRUCTIONS: Please relax yourself.")
    else:
        print("INSTRUCTIONS: Please concentrate intently on a task.")
    
    while True:
        ready = input("Are you ready to begin? (y/n): ").lower()
        if ready == 'y':
            return True
        elif ready == 'n':
            print("Session Aborted")
            return False
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

# collect data from the brainwave headset
def run_collection_session():
    scores = []
    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
            print(f"\nSuccessfully connected to Arduino on {SERIAL_PORT}.")
            ser.flushInput()

            print("Starting data collection in 3... 2... 1... GO!")
            start_time = time.time()
            
            while time.time() - start_time < COLLECTION_DURATION:
                remaining_time = COLLECTION_DURATION - (time.time() - start_time)
                line = ser.readline().decode('utf-8', errors='ignore').strip()

                if "Attation" in line:
                    signal_match = re.search(r'SignalQuality: (\d+)', line)
                    attention_match = re.search(r'Attation: (\d+)', line) 

                    if signal_match and attention_match:
                        signal_quality = int(signal_match.group(1))
                        if signal_quality == 0:
                            attention_value = int(attention_match.group(1))
                            scores.append(attention_value)
                            print(f"Time left: {remaining_time:02.0f}s | Signal: GOOD | Attention: {attention_value:3d} | Number of data points collected: {len(scores)}", end='\r')
                        else:
                            print(f"Time left: {remaining_time:02.0f}s | Signal: BAD ({signal_quality}) | Skipping that data point due to bad signal...", end='\r')
            
            print("\n" + "="*60)
            print("Session Complete!")

    except serial.SerialException:
        print(f"\nFATAL ERROR: Could not open serial port '{SERIAL_PORT}'.")
        print("Check if Arduino is plugged in, port is correct, and Serial Monitor is closed.")
        return None
    
    return scores

def main():
    print("--- Brainwave Data Collection---")
    while True:
        participant_id = input("\nEnter a unique Participant ID: ").strip()
        if not participant_id:
            print("Participant ID cannot be empty.")
            continue

        if get_user_confirmation("relaxed", participant_id):
            relaxed_scores = run_collection_session()
            if relaxed_scores and len(relaxed_scores) > 10:
                filename = f"relaxed_{participant_id}.txt"
                with open(filename, "w") as f:
                    for score in relaxed_scores: f.write(f"{score}\n")
                print(f"Collected {len(relaxed_scores)} data points. Saved to {filename}")
            else:
                print("Failed to collect sufficient relaxed data. Please try again.")
                continue

        if get_user_confirmation("concentrating", participant_id):
            concentrating_scores = run_collection_session()
            if concentrating_scores and len(concentrating_scores) > 10:
                filename = f"concentrating_{participant_id}.txt"
                with open(filename, "w") as f:
                    for score in concentrating_scores: f.write(f"{score}\n")
                print(f"Collected {len(concentrating_scores)} data points. Saved to {filename}")
            else:
                print("Failed to collect sufficient concentrating data. Please try again.")
                continue
        
        another = input("\nDo you want to collect data for another participant? (y/n): ").lower()
        if another != 'y':
            break
            
    print("\nData collection step is now complete.")

if __name__ == '__main__':
    main()