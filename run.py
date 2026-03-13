"""Entry point to start the Federated Learning server."""

import uvicorn


def main():
    print("=" * 60)
    print("  Federated Learning - Diabetic Retinopathy Detection")
    print("  Starting server at http://localhost:8000")
    print("=" * 60)
    print()
    print("Open http://localhost:8000 in your browser to access the dashboard.")
    print()

    uvicorn.run(
        "federated.server.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
