## DEPLOY FRONTEND

### Deploy the frontend to Azure Static Web Apps

    - Fork the github repo or create new
    - ensure frontend/next.config.ts contains ```output: "export"```
    - Commit and push the code to Github
    - Go to Azure Portal -> Static Web Apps → Create
    - Create or choose a resource group, Enter name, region, plan (choose free)
    - Deployment Source: Github, select repository and branch
    - App Location as "/frontend", leave api_location empty, output_location as "out"
    - Click Review and Create. 
    - An Azure Workflow file will be created in .github/workflows inside your github repo.
    - This workflow file will run automatically, and you can view the site available via the link on azure portal.
    - The workflow file is still on github. You need to do git pull on local.
    - This workflow runs everytime you commit your changes on local and push to github.
    - Currently the frontend will point to localhost, so run the backend on your local.
    - Once your api is hosted, you can edit the workflow file to point to your API
    ```
      - name: Build And Deploy
        id: builddeploy
        uses: Azure/static-web-apps-deploy@v1
        with:
          azure_static_web_apps_api_token: ${{ secrets.AZURE_STATIC_WEB_APPS_API_TOKEN_BLACK_MUSHROOM_04AABD610 }}
          repo_token: ${{ secrets.GITHUB_TOKEN }} # Used for Github integrations (i.e. PR comments)
          action: "upload"
          ###### Repository/Build Configurations - These values can be configured to match your app requirements. ######
          # For more information regarding Static Web App workflow configurations, please visit: https://aka.ms/swaworkflowconfig
          app_location: "/frontend" # App source code path
          api_location: "" # Api source code path - optional
          output_location: "out" # Built app content directory - optional
          ###### End of Repository/Build Configurations ######
        env:
          NEXT_PUBLIC_API_URL: "https://your-api.com"
    ```
    - Then push your changes, and the GitHub Actions workflow for azure will run automatically. Now your Frontend will point to your API.

----------------------------------------------------------------

## DEPLOY BACKEND API

### Local

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

---------------------------------------------------------------

### Azure Container apps:

    Step 1: 
        push latest docker image to dockerhub (using github actions)

    Step 2: In Azure Portal
        Create a Resource Group, choose a Region (e.g., East US).

    Step 3: Create Persistent Storage for ChromaDB
        This Azure File Share will act as the persistent disk for your container.
        3.1 Create the Storage Account, same resource group and region.
            Choose config for Azure file share, LRS, choose HDD (pay as you go), (fast), (premium ssd is not free)
        3.2. Create the File Share
            On the left-hand menu of your storage account page, find and click on File shares. (choose transaction optimized)

    Step 4: Uploading data
        Option 1: upload directly
            storage account -> select from menu: data storage -> select file share -> browse -> upload using browser
        Option 2: azure storage explorer
            download azure storage explorer.
            Sign in, you will see two tenants (click the default directory one, and click Open Explorer)
            Now click Azure Subscription -> storage acc -> file share.
            Upload folder

    Step 5: Create Container App Environment
        create env, with same region

    Step 6: Create the Azure Container App
            Select resource group, env etc.
            Select dockerhub registry and enter the image name.
            Select 1.0 vCPU / 2.0 Gi memory. You can adjust this later.
            Add env variables manually
            Ingress Tab -> choose HTTP, source anywhere, target port 8000 (gunicorn port), leave insecure connection unchecked.

            Review and Create

    Step 7: Add volume to container:

        Go to the Container App env
            settings -> azure files -> add -> use smb, enter details of your file share, storage account, access - read/write
        Go to Container Apps:
            Application -> Volumnes -> Create new volume -> choose existing volume, same name.
                This will create a new revision
            Revision and Replicas:
                Create new revision
                give a suffix for new name
                Containers list -> checkbox container -> Edit Container -> volume mounts -> select volume, and mount path (/app/processed_data)
                Click Create

    Step 8: Test
        open the url of the container app (or the revision url). API should be accesible.


--------------------------------------------------------------------------

### Azure (VM + fileshare/managed disk: run docker conatiner in VM):

    1. Create VM
            keep default settings. choose a vm with free tier eligibility, selected by default.
            you will recieve ssh key while creating. save the ssh key safely.
    
    2. connect to VM via SSH:
        4.1 get the public ip of vm from vm overview
        4.2 get vm admin username from vm settings -> operating system
        4.3 change permission of .pem ssh key, to allow only current windows user access to file. (take help from LLM)
        4.4 open powershell:
            ssh -i path\to\your\key.pem username@dnsname
    
    3. Setup docker on vm: [ONLY ONCE]
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
            sudo apt install cifs-utils

    [STORAGE]

        4. [File Share]:
            4.1 Create Storage account in VM region, and same resource group as vm
                    Choose config for Azure file share, LRS, choose HDD (pay as you go), (fast), (premium ssd is not free)
            4.2 Create the File Share
                    On the left-hand menu of your storage account page, find and click on File shares. (choose transaction optimized)
            4.3 Mount to VM:
                go to storage account - fileshare -> click connect -> copy script for linux.
                In VM, paste the script in ~/fileshare-setup.sh, change file permissions from 0755 to 0777.
                Also check if the script makes fstab entry. If not make it manually. (to work even after vm reboot)
                Run the script:
                    sudo chmod +x fileshare-setup.sh
                    ./fileshare-setup.sh
                Verify: 
                    df -h

            4.4: Uploading data
                Option 1: upload directly
                    storage account -> select from menu: data storage -> select file share -> browse -> upload using browser
                Option 2: azure storage explorer
                    download azure storage explorer.
                    Sign in, you will see two tenants (click the default directory one, and click Open Explorer)
                    Now click Azure Subscription -> storage acc -> file share.
                    Upload folder

        OR

        5 [Managed Disk]
    
            5.1. Create Managed Disk in the same region as vm
                select size as p6: 64GB, as thats covered under free tier
            5.2. Configure VM for managed disk:
                Go the VM settings and attach the managed disk. enable read only caching.
            5.3 Prepare Managed Disk for data [DO ONLY once]
                (connect to vm via ssh)
                5.3.1 Check the disks available: 
                    execute command: lsblk
                    check your disk should be "sdc" most probably. [CAUTION]: CHANGE BELOW COMMAND IF IT IS SOMETHING ELSE.
                5.3.2 format disk:
                    sudo mkfs.ext4 /dev/sdc
                5.3.3 Create data directory, where the disk will be mounted:
                    sudo mkdir /data
                5.3.4 mount disk:
                    sudo mount /dev/sdc /data
                5.3.5 Make mount permanent (disk survives vm reboot):
                    get uuid:
                        sudo blkid /dev/sdc
                    open vim and edit the /etc/fstab file:
                        sudo vim /etc/fstab
                    enter this line at the last: UUID=<the-uuid-you-got-from-blkid>  /data  ext4  defaults,nofail  0  2
                    save: :wq
    
            5.4. upload data from pc to VM managed disk 
                In VM (connect to vm via ssh)
                    give permission:
                        sudo chown $USER:$USER /data
                    create folder inside /data
                        sudo mkdir -p /data/processed_data
                        sudo chown -R $USER:$USER /data/processed_data
                
                In Powershell from project root [WHENEVER DATA CHANGES]:
                    copy data from local to vm using scp:
                        scp -i path\to\your\key.pem -r ./processed_data your-user@your-vm-ip:/data/


    7. setup nginx server for http and https access:
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

    8. enable tls for httpps:
        sudo apt-get install -y certbot python3-certbot-nginx
        # run certbot for your domain to setup certificate
        sudo certbot --nginx -d your_azure_dns_name.cloudapp.azure.com
        
        access website at https://your_azure_dns_name


    9. Setup Github CI/CD:
        In local project: 
            create file .github/workflows/docker-publish.yml  (take help from llm to publish to dockerhub - manual trigger)
            push to github
        In github.com:
            go to repo -> settings -> actions -> add repo secret.
                Add DOCKERHUB_USERNAME, DOCKERHUB_TOKEN
            go to repo -> github actions and run the workflow:
                it will build and push image to dockerhub

    10. clone repo:
        sudo git clone https://github.com/username/repo_name.git
        cd repo_name

    11. create a new .env file using nano, and paste values
        set DATA_HOST_PATH as per managed_disk/fileshare  [/data/processed_data or /media/filesharename]

    
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

                # to check container logs:
                    sudo docker logs --since 10m documentation-agent-container

                # connect to running container:
                    sudo docker exec -it documentation-agent-container /bin/bash

        3. If write permission error. host may have different OWNER for bind mount folder and /data/processed folder.
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
            

### Observations:
    1. building docker image on Vm with just 1GB RAM, it occupied all RAM, and slowed down like stuck. The image is just 2.6GB.
    -> Stick with github Actions to build and publish images to dockerhub

    2. deploying as container running on VM, 1gb ram is insufficient to run chromadb, api crashes and restarts with out of memory error, whenever chromadb retrieval is used. Database was less than 100MB.
    -> if running chromadb on vm - Need atleast 2GB RAM.
    -> use online vector store like pinecone, supabase -> 1GB ram is sufficient
    
    3. read performance for keyword retrieval (bm25) is almost same on file share and managed disk.
        -> use file share, its simpler, and cheaper. managed disk has only slight performance benefit considering read.
