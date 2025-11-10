#pragma once
#include "../../core/compiler.hpp"
#include "../../core/types.hpp"
#include <cstring>
#include <netinet/ether.h>
#include <netinet/ip.h>
#include <netinet/udp.h>

namespace ultra::network {

#pragma pack(push, 1)

struct EthernetHeader {
    uint8_t dst_mac[6];
    uint8_t src_mac[6];
    uint16_t ethertype;
};

struct IPv4Header {
    uint8_t  version_ihl;
    uint8_t  tos;
    uint16_t total_length;
    uint16_t id;
    uint16_t flags_offset;
    uint8_t  ttl;
    uint8_t  protocol;
    uint16_t checksum;
    uint32_t src_ip;
    uint32_t dst_ip;
};

struct UDPHeader {
    uint16_t src_port;
    uint16_t dst_port;
    uint16_t length;
    uint16_t checksum;
};

#pragma pack(pop)

/**
 * Zero-copy Ethernet/IP/UDP parser
 * Parses in-place, no allocations
 */
class EthernetParser {
public:
    struct ParsedPacket {
        const uint8_t* payload;
        uint16_t payload_len;
        uint32_t src_ip;
        uint32_t dst_ip;
        uint16_t src_port;
        uint16_t dst_port;
        Timestamp timestamp_ns;
        bool valid;
    };
    
    ULTRA_ALWAYS_INLINE static ParsedPacket parse(
        const uint8_t* packet, 
        size_t packet_len,
        Timestamp timestamp_ns
    ) noexcept;
    
    ULTRA_ALWAYS_INLINE static uint16_t ntohs_fast(uint16_t n) noexcept {
        return __builtin_bswap16(n);
    }
    
    ULTRA_ALWAYS_INLINE static uint32_t ntohl_fast(uint32_t n) noexcept {
        return __builtin_bswap32(n);
    }
};

} // namespace ultra::network
