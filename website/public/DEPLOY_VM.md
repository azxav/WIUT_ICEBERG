# Deploy ICEBERG on one cloud VM

This runs the frontend and FastAPI demo together on one Linux VM. Caddy is the only public service; it handles HTTPS and forwards requests to the web container. The web container serves the built React site and proxies `/api` to the private API container. No hosted app platform, GitHub repository, or GitHub-triggered deployment is required.

## VM prerequisites

- A Linux VM with Docker Engine and the Docker Compose plugin.
- A DNS name whose A/AAAA record points to the VM's public IP.
- In the cloud firewall and VM firewall, allow inbound TCP ports 80 and 443 and UDP port 443. With public DNS pointing to the VM, Caddy automatically issues and renews HTTPS certificates ([Caddy documentation](https://caddyserver.com/docs/automatic-https)).
- The project source, including `weights/yolo11s.pt`, transferred to the VM over SSH/SCP or another direct file-transfer method.

The VM must have enough CPU and memory for PyTorch and video inference. Select its size based on the expected number and length of concurrent uploads.

## Start the deployment

From the project root on the VM:

```sh
cp .env.example .env
```

Edit `.env` and set `DOMAIN` to the DNS name pointing at the VM. Then build and start all three containers:

```sh
docker compose up -d --build
docker compose ps
```

Check the public site and API health endpoint at `https://YOUR_DOMAIN/` and `https://YOUR_DOMAIN/health`. To inspect startup or runtime logs:

```sh
docker compose logs -f caddy web api
```

The API has no published host port. Browser uploads and status polling use the same HTTPS origin as the site. Only Caddy binds public ports; Nginx sends `/api` and `/health` requests to the API on Docker's private network.

## Update the site

Transfer the changed source files to the VM, then rebuild and restart:

```sh
docker compose up -d --build
```

Uploaded videos are temporary and deleted after processing. Job state is held in memory, so restarting the API clears active jobs. Caddy's named volumes preserve its TLS certificates across container replacement.

## Local development

Run the API on port 8000 from the repository root. Run `npm ci` and `npm run dev` in `website/`; Vite forwards `/api` requests to the API. Production also uses the same-origin `/api` path, routed through Nginx on the VM.

This repository contains the deployment stack and instructions. A live deployment still needs a provisioned VM, DNS name, firewall access, and an authorized deploy session.
