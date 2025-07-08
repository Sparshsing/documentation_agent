Local:
    build image:
        docker compose build
    run container:
        docker compose up -d
    stop container:
        docker compose down --remove-orphans
        Then also manually remove container using docker desktop, if still not deleted.
    debug container:
        # create debug service in docker-ccompose file
        docker compose up -d api-debug
        Then open vs code/cursor:
            Cursor:
                go to debug tab, expand the debug/run dropdown. select add configuration.
                select python debugger. then select localhost, then select port 5678. 
                In the launch.json file, change "remoteRoot" to /app
                put breakpoints and start.
            Vs code:
                go to debug tab, click create a launch json file.
                select python debugger. then select localhost, then select port 5678. 
                In the launch.json file, change "remoteRoot" to /app
                put breakpoints and start.

--------------------------------------------------------------------------
Azure (VM + managed disk: run docker conatiner in VM):
    1. Create VM
            keep default settings. choose a vm with free tier eligibility, selected by default.
            you will recieve ssh key while creating. save the ssh key safely.
    2. Create Managed Disk in the same region as vm
        select size as p6: 64GB, as thats covered under free tier
    3. Configure VM for managed disk:
        3.1. Go the VM settings and attach the managed disk. enable read only caching.
    4. connect to VM via SSH:
        4.1 get the public ip of vm from vm overview
        4.2 get vm admin username from vm settings -> operating system
        4.3 change permission of .pem ssh key, to allow only current windows user access to file. (take help from LLM)
        4.4 open powershell:
            ssh -i path\to\your\key.pem username@public_ip_address
    5. Prepare Managed Disk for data [DO ONLY once]
        (connect to vm via ssh)
        5.1 Check the disks available: 
            execute command: lsblk
            check your disk should be "sdc" most probably. [CAUTION]: CHANGE BELOW COMMAND IF IT IS SOMETHING ELSE.
        5.2 format disk:
            sudo mkfs.ext4 /dev/sdc
        5.3 Create data directory, where the disk will be mounted:
            sudo mkdir /data
        5.4 mount disk:
            sudo mount /dev/sdc /data
        5.5 Make mount permanent (disk survives vm reboot):
            get uuid:
                sudo blkid /dev/sdc
            open vim and edit the /etc/fstab file:
                sudo vim /etc/fstab
            enter this line at the last: UUID=<the-uuid-you-got-from-blkid>  /data  ext4  defaults,nofail  0  2
            save: :wq
    
    6. upload data from pc to VM managed disk 
        In VM (connect to vm via ssh)
            give permission:
                sudo chown $USER:$USER /data
            create folder inside /data
                sudo mkdir -p /data/processed_data
                sudo chown -R $USER:$USER /data/processed_data
        
        In Powershell from project root [WHENEVER DATA CHANGES]:
            copy data from local to vm using scp:
                scp -i path\to\your\key.pem -r ./processed_data your-user@your-vm-ip:/data/
            

    7. Setup docker on vm: [ONLY ONCE]
        follow: https://docs.docker.com/engine/install/ubuntu/
        Run below commands:
            # Add Docker's official GPG key:
            sudo apt-get update
            sudo apt-get install ca-certificates curl
            sudo install -m 0755 -d /etc/apt/keyrings
            sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
            sudo chmod a+r /etc/apt/keyrings/docker.asc

            # Add the repository to Apt sources:
            echo \
            "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
            $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
            sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
            sudo apt-get update

            sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

            sudo apt-get install -y nginx


    9. clone repo:
        sudo git clone https://github.com/Sparshsing/documentation_agent.git
        cd documentation_agent

    9. create a new .env file using nano, and paste values

    10. Github CI/CD:
            create token: read, write delete
            
            configure secrets in github Actions:
                go to repo -> settings -> actions -> add repo secret.
                Add for DOCKERHUB_USERNAME, DOCKERHUB_TOKEN

                in project, make file .github/workflows/docker-publish.yml
                go to github actions and run the workflow:
                    it will build and push image to dockerhub


    11: setup nginx server for http and https access
        a. sudo nano /etc/nginx/sites-available/fastapi_proxy
        b. paste this:
            # /etc/nginx/sites-available/fastapi_proxy
            server {
                listen 80;
                server_name your_azure_dns_name.cloudapp.azure.com; # Replace with your domain(s)

                location / {
                    proxy_pass http://127.0.0.1:8000;
                    proxy_set_header Host $host;
                    proxy_set_header X-Real-IP $remote_addr;
                    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
                    proxy_set_header X-Forwarded-Proto $scheme;
                }
            }

        c. symbolic link 
            sudo ln -s /etc/nginx/sites-available/fastapi_proxy /etc/nginx/sites-enabled/
            # It's good practice to remove the default site if you're not using it
            sudo rm /etc/nginx/sites-enabled/default

        d. test nginx
            sudo nginx -t
        
        e. if ok, restart:
            sudo systemctl restart nginx

        f. Open port 80, 443 for http, htpps:
            azure portal -> vm -> networking -> add inbound rule -> defaults, select service http, done.
            similarly for https

    12. enable tls for httpps:
        sudo apt-get install -y certbot python3-certbot-nginx
        # run certbot for your domain to setup certificate
        sudo certbot --nginx -d your_azure_dns_name.cloudapp.azure.com
        
        access website at https://your_azure_dns_name



    On every code change / First run
        1. github:
                push changes from local
                start github action: to build and publish image
        2. In VM:
                cd documentation_agent
                git pull
                sudo docker compose pull

                # Recreate the container with the new image
                sudo docker compose up -d

                # if need to stop:
                    sudo docker compose down --remove-orphans

        3. If permission error. host may have different OWNER for bind mount folder and /data/processed folder.
            # fixed by entrypoint.sh
            If not chainging uid and GID of app user in entrypoint.sh, then:
                # add permission for app user (defined in docker compose):
                # find container user id and add it as 
                # actually ignore these. Ask an LLM for help.
                # sudo chown -R 100:101 /data/processed_data
                # sudo chmod -R 775 /data/processed_data


    connect to vm via ssh (from powershell in project root):
    ssh -i documentation-agent-vm-ssh-key.pem azureuser@your_azure_dns_name.cloudapp.azure.com

    connect to running container:
        docker exec -it --user app documentation-agent-container bash
        or sudo docker exec -it documentation-agent-container /bin/bash
            

    Observations:
        1. building docker image on Vm with just 1GB RAM, it occupied all RAM, and slowed down like stuck. The image is just 2.6GB.
        -> Stick with github Actions to build and publish images to dockerhub

        2. deploying as container running on VM, 1gb ram is insufficient, api crashes and restarts with out of memory error, whenevr chromadb retrieval is used. Database was less than 100MB.
        -> Need atleast 2GB RAM.









