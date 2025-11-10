#include "ultra/market-data/itch/decoder.hpp"
#include <iostream>

namespace ultra::md::itch {

ITCHDecoder::ITCHDecoder() {
    // We could pre-populate the symbol table from a config file here
    // For now, it's manual via register_symbol
}

void ITCHDecoder::register_symbol(const char* symbol, SymbolId id) {
    uint64_t symbol_int = char8_to_uint64(symbol);
    uint32_t index = hash_symbol(symbol_int);
    
    // Simple linear probe
    for(size_t i = 0; i < SYMBOL_HASH_SIZE; ++i) {
        uint32_t current_index = (index + i) & (SYMBOL_HASH_SIZE - 1);
        if (symbol_table_[current_index].id == INVALID_SYMBOL) {
            symbol_table_[current_index].symbol_int = symbol_int;
            symbol_table_[current_index].id = id;
            return;
        }
    }
    // Error: hash table full
    std::cerr << "ITCHDecoder Error: Symbol hash table full." << std::endl;
}

SymbolId ITCHDecoder::lookup_symbol(const char* symbol) const noexcept {
    uint64_t symbol_int = char8_to_uint64(symbol);
    uint32_t index = hash_symbol(symbol_int);
    
    for(size_t i = 0; i < SYMBOL_HASH_SIZE; ++i) {
        uint32_t current_index = (index + i) & (SYMBOL_HASH_SIZE - 1);
        if (symbol_table_[current_index].symbol_int == symbol_int) {
            return symbol_table_[current_index].id;
        }
        if (symbol_table_[current_index].id == INVALID_SYMBOL) {
            return INVALID_SYMBOL;
        }
    }
    return INVALID_SYMBOL;
}

ULTRA_HOT ULTRA_ALWAYS_INLINE 
ITCHDecoder::DecodedMessage ITCHDecoder::decode(const uint8_t* data, size_t len, Timestamp rdtsc_ts) noexcept {
    DecodedMessage msg{};
    msg.tsc = rdtsc_ts;
    msg.valid = false;
    msg.event_type = MDEventType::UNKNOWN;
    
    if (ULTRA_UNLIKELY(len < sizeof(MessageHeader))) return msg;
    
    const auto* header = reinterpret_cast<const MessageHeader*>(data);
    const auto msg_type = static_cast<MessageType>(header->type);

    // This switch is the critical path
    switch(msg_type) {
        case MessageType::ADD_ORDER: {
            if (ULTRA_UNLIKELY(len < sizeof(AddOrder))) return msg;
            const auto* add = reinterpret_cast<const AddOrder*>(data);
            
            msg.event_type = MDEventType::ADD_ORDER;
            msg.exchange_ts = bswap_64(add->timestamp);
            msg.order_id = bswap_64(add->order_ref_number);
            msg.side = (add->buy_sell_indicator == 'B') ? Side::BUY : Side::SELL;
            msg.quantity = bswap_32(add->shares);
            msg.price = decode_price(add->price);
            msg.symbol_id = lookup_symbol(add->stock);
            msg.valid = true;
            return msg;
        }
        
        case MessageType::ORDER_DELETE: {
            if (ULTRA_UNLIKELY(len < sizeof(OrderDelete))) return msg;
            const auto* del = reinterpret_cast<const OrderDelete*>(data);
            
            msg.event_type = MDEventType::DELETE_ORDER;
            msg.exchange_ts = bswap_64(del->timestamp);
            msg.order_id = bswap_64(del->order_ref_number);
            msg.valid = true;
            return msg;
        }

        case MessageType::ORDER_REPLACE: {
            if (ULTRA_UNLIKELY(len < sizeof(OrderReplace))) return msg;
            const auto* rep = reinterpret_cast<const OrderReplace*>(data);
            
            msg.event_type = MDEventType::MODIFY_ORDER;
            msg.exchange_ts = bswap_64(rep->timestamp);
            msg.order_id = bswap_64(rep->original_order_ref);
            msg.new_order_id = bswap_64(rep->new_order_ref);
            msg.quantity = bswap_32(rep->shares);
            msg.price = decode_price(rep->price);
            msg.valid = true;
            return msg;
        }
        
        // ... Add other cases: ORDER_EXECUTED, TRADE, etc.
        
        default:
            // Not a message type we care about for book building
            return msg;
    }
}

} // namespace ultra::md::itch
