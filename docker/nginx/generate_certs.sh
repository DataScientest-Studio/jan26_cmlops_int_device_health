#!/bin/bash
set -e

# Generate self-signed SSL certificates for local development
# For production, use Let's Encrypt or your organization's CA

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSL_DIR="${SCRIPT_DIR}/ssl"

echo "🔐 Generating self-signed SSL certificates..."
echo "   Location: ${SSL_DIR}"

# Create SSL directory if it doesn't exist
mkdir -p "${SSL_DIR}"

# Generate private key and certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout "${SSL_DIR}/key.pem" \
    -out "${SSL_DIR}/cert.pem" \
    -subj "/C=US/ST=State/L=City/O=MLOps/OU=DeviceHealth/CN=localhost" \
    -addext "subjectAltName=DNS:localhost,DNS:*.localhost,IP:127.0.0.1"

# Set appropriate permissions
chmod 600 "${SSL_DIR}/key.pem"
chmod 644 "${SSL_DIR}/cert.pem"

echo "✅ Certificates generated successfully!"
echo ""
echo "Files created:"
echo "  - ${SSL_DIR}/cert.pem (public certificate)"
echo "  - ${SSL_DIR}/key.pem (private key)"
echo ""
echo "📝 Next steps:"
echo "  1. Certificates are ready to use"
echo "  2. HTTPS server is configured in nginx.conf"
echo "  3. docker-compose.yml mounts certificates automatically"
echo "  4. Restart nginx: docker compose restart nginx"
echo ""
echo "⚠️  Note: Browser will show security warning (self-signed cert)"
echo "   This is expected for local development. Click 'Advanced' → 'Proceed'"
echo ""
echo "🔍 Test HTTPS:"
echo "   curl -k https://localhost/health"
echo "   open https://localhost/docs"
