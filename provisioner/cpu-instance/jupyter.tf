variable "instance_type" {}
variable "project_directory" {}
variable "region" {}
variable "vpc_id" {}
variable "spot_price" {}
variable "number_of_instances" {}

provider "aws" {
  region = "${var.region}"
  profile = "default"
}

data "aws_ami" "ubuntu" {
  most_recent = true

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-trusty-14.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  owners = ["099720109477"]
}

resource "aws_security_group" "allow_ssh" {
  name        = "allow_ssh"
  description = "Allow all inbound ssh connections"
  vpc_id = "${var.vpc_id}"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 8888
    to_port     = 8888
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port       = 0
    to_port         = 0
    protocol        = "-1"
    cidr_blocks     = ["0.0.0.0/0"]
  }
}

resource "aws_spot_instance_request" "jupyter" {
  spot_price = "${var.spot_price}"
  count = "${var.number_of_instances}"

  ami = "${data.aws_ami.ubuntu.id}"
  instance_type = "${var.instance_type}"
  key_name = "spark_key"
  security_groups = ["${aws_security_group.allow_ssh.name}"]

  ebs_block_device {
    volume_type = "gp2"
    volume_size = "50"
    delete_on_termination = "true"
    device_name = "/dev/sda1"
  }

  connection {
        user = "ubuntu"
        private_key = "${file("~/.ssh/spark_key.pem")}"
  }

  tags {
    Name = "jupyter"
  }

  provisioner "file" {
    source = "${var.project_directory}"
    destination = "/home/ubuntu/toxic-comment-classification/codebase/"
  }

  provisioner "file" {
    source      = "setup.sh"
    destination = "/home/ubuntu/setup.sh"
  }

  provisioner "remote-exec" {
    inline = [
      "chmod +x /home/ubuntu/setup.sh",
      "./setup.sh"
    ]
  }
}

output "ssh-and-forward-jupyter-notebook-port" {
  value = "ssh -i ~/.ssh/spark_key.pem ubuntu@${aws_spot_instance_request.jupyter.public_ip} -L 8888:localhost:8888"
}
