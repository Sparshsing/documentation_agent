#!/bin/sh

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the path to your data directory inside the container
DATA_DIR="/app/processed_data"

# Get the UID and GID of the data directory's owner
# This will be the UID/GID from the host machine
TARGET_UID=$(stat -c "%u" "$DATA_DIR")
TARGET_GID=$(stat -c "%g" "$DATA_DIR")

echo "--- [Entrypoint] ---"
echo "Starting container setup..."
echo "Host volume owner is UID=${TARGET_UID} GID=${TARGET_GID}"

# Get the current UID/GID of the 'app' user inside the container
CURRENT_UID=$(id -u app)
CURRENT_GID=$(id -g app)

echo "Container's 'app' user is currently UID=${CURRENT_UID} GID=${CURRENT_GID}"

# If the container's 'app' user ID does not match the host's volume owner ID, change it.
if [ "$CURRENT_UID" != "$TARGET_UID" ] || [ "$CURRENT_GID" != "$TARGET_GID" ]; then
    echo "UID/GID mismatch detected. Synchronizing..."

    # Change the GID of the 'app' group first
    groupmod -o -g "$TARGET_GID" app
    # Change the UID of the 'app' user
    # The -o flag allows using a non-unique UID
    usermod -o -u "$TARGET_UID" app

    echo "User 'app' is now UID=$(id -u app) GID=$(id -g app)"
else
    echo "UID/GID already in sync. No changes needed."
fi

echo "Setup complete. Dropping privileges and starting the application..."
echo "--- [End Entrypoint] ---"

# Use gosu to drop from root to the 'app' user,
# then execute the command passed to this script (your CMD).
# "$@" is a special variable that holds all arguments passed to the script.
exec gosu app "$@"