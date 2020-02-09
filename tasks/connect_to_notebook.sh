#!/bin/bash

ssh -i ~/free_tier.pem -L 8000:localhost:8889 ec2-user@ec2-18-188-64-16.us-east-2.compute.amazonaws.com
