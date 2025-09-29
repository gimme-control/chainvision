# Inspiration
When violent crimes occur, every second counts. We wanted to build something that helps law enforcement react faster by using technology that many cities already have, security cameras. Our goal was to transform passive surveillance into an active early-warning and suspect-tracking system.

# What it does
ChainVision instantly recognizes weapons in security footage, locks onto the person holding them, and tracks their movement across multiple cameras. At the same time, it generates a professional, law–enforcement–grade description of the suspect so officers get actionable details within seconds.

# How we built it
We trained a YOLO model to detect guns and knives, then used OpenCV to visualize and track the suspect across camera feeds. For person re-identification, we generated vector embeddings for each individual, stored them in a SQLite database, and used cosine similarity to match and re-identify the same person across different cameras. From the captured snapshots, we called the Gemini API to produce a concise, police-style description based on the Chicago PD’s public instructions for how to describe a suspect. The entire pipeline operates in real time, creating a seamless chain of visual evidence.

# Challenges we ran into
Tracking a suspect between different camera feeds was by far the hardest part. OpenCV is built to process frames one at a time, so maintaining a single identity across multiple cameras isn’t native to the library. We dove into academic papers on person re-identification (ReID) and adapted those techniques to our pipeline, experimenting with different models and similarity metrics until we could reliably match the same suspect across separate angles, all while keeping latency low.

# Accomplishments that we're proud of
We’re proud that ChainVision produces law-enforcement–actionable intelligence, not just computer-vision output. Our suspect descriptions follow the Chicago Police Department’s official guidelines for what details officers need, ensuring the information is both accurate and valuable. We also successfully integrated person re-identification (ReID) to track suspects across multiple cameras and reliably map each detected weapon to the individual carrying it, which was a key step toward making the system practical for real-world response.

# What we learned
We deepened our understanding of custom-class training in YOLO, real-time video processing with OpenCV, and prompt engineering for image-based LLMs. We also learned how critical latency and pipeline optimization are when building live safety systems.

# What's next for ChainVision
From a technology perspective, we would like to broaden our detection and move to vehicle crime, theft, and other criminal activity. From a business perspective, we want to explore partnerships with city agencies and police departments to pilot ChainVision in controlled environments.
