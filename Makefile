all:
	python -m grpc_tools.protoc --proto_path=./service/ --python_out=./service/ --grpc_python_out=./service/ ./service/*.proto

clean:
	rm -f ./service/*_pb2*.py
	rm -f ./service/*.pyc
