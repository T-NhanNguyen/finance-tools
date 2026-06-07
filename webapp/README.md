# webapp

## Start
```
cp .env.example .env                                                                                                                                              
docker compose up -d
cloudflared tunnel --url http://localhost:8000
```

## Stop
```
docker compose down
```

## Clean orphans
```
docker compose down --remove-orphans
docker system prune -f
```
