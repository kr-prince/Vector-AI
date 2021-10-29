# Vector-AI
This repo will have all the code to solve Vector.AI challenge problems.



Clone the zip repo locally. Run `conda env create -f environment.yml` 


Download latest Kafka binary from https://kafka.apache.org/downloads and extract it.

Ensure JRE and JDK are installed and Java environment variables are set. To check, open cmd or terminal and try 
java --version and javac --version

From inside the kafka folder, start the zookeper server using the following commands
bin/zookeeper-server-start.sh config/zookeeper.properties

