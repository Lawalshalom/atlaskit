"""`python -m atlaskit_gateway` — start the gateway."""
import os

import uvicorn


def main() -> None:
    uvicorn.run("atlaskit_gateway.app:app", host=os.environ.get("HOST", "0.0.0.0"),
                port=int(os.environ.get("PORT", "8080")), proxy_headers=True)


if __name__ == "__main__":
    main()
