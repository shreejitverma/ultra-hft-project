#include "ultra/network/parsers/ethernet_parser.hpp"

namespace ultra::network {

ULTRA_ALWAYS_INLINE EthernetParser::ParsedPacket EthernetParser::parse(
    const uint8_t* packet, 
    size_t packet_len,
    Timestamp timestamp_ns
) noexcept {
    ParsedPacket result = {nullptr, 0, 0, 0, 0, 0, timestamp_ns, false};
    
    // Min packet size: Eth + IP + UDP
    if (ULTRA_UNLIKELY(packet_len < (sizeof(EthernetHeader) + sizeof(IPv4Header) + sizeof(UDPHeader)))) {
        return result;
    }
    
    const auto* eth_hdr = reinterpret_cast<const EthernetHeader*>(packet);
    const uint8_t* ip_payload = packet + sizeof(EthernetHeader);

    // Check for IPv4
    if (ULTRA_LIKELY(eth_hdr->ethertype == ntohs_fast(ETHERTYPE_IP))) {
        const auto* ip_hdr = reinterpret_cast<const IPv4Header*>(ip_payload);
        
        // Basic IP header validation
        if (ULTRA_UNLIKELY((ip_hdr->version_ihl & 0xF0) != 0x40)) {
            return result; // Not IPv4
        }
        
        // Check for UDP
        if (ULTRA_LIKELY(ip_hdr->protocol == IPPROTO_UDP)) {
            uint8_t ip_header_len = (ip_hdr->version_ihl & 0x0F) * 4;
            const auto* udp_hdr = reinterpret_cast<const UDPHeader*>(ip_payload + ip_header_len);
            
            result.payload = ip_payload + ip_header_len + sizeof(UDPHeader);
            result.payload_len = ntohs_fast(udp_hdr->length) - sizeof(UDPHeader);
            
            // Check for packet truncation
            if (ULTRA_UNLIKELY(result.payload + result.payload_len > packet + packet_len)) {
                return result; // Invalid length
            }
            
            result.src_ip = ip_hdr->src_ip; // Already network byte order
            result.dst_ip = ip_hdr->dst_ip; // Already network byte order
            result.src_port = udp_hdr->src_port; // Already network byte order
            result.dst_port = udp_hdr->dst_port; // Already network byte order
            result.valid = true;
            
            return result;
        }
    }
    // Note: Skipping VLAN (802.1Q) parsing for simplicity in this stub
    
    return result;
}

} // namespace ultra::network
