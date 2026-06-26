class ContiguousAllocator:
    """Allocates a single contiguous block per request at max sequence length."""
    def __init__(self, total_memory_mb):
        # Track memory as a list of (start, end) free regions
        self.free_regions = [(0, total_memory_mb)]
        self.allocations = {}  # request_id -> (start, end)
    
    def allocate(self, request_id, size_mb):
        """Find first contiguous region that fits."""
        for i, (start, end) in enumerate(self.free_regions):
            if end - start >= size_mb:
                self.allocations[request_id] = (start, start + size_mb)
                # Split the free region
                remaining = []
                if start + size_mb < end:
                    remaining.append((start + size_mb, end))
                self.free_regions = self.free_regions[:i] + remaining + self.free_regions[i+1:]
                return True
        return False  # No contiguous region large enough
    
    def free(self, request_id):
        if request_id in self.allocations:
            freed = self.allocations.pop(request_id)
            self.free_regions.append(freed)
            self.free_regions.sort()
            # Merge adjacent regions
            self._merge()
    
    def _merge(self):
        merged = [self.free_regions[0]]
        for start, end in self.free_regions[1:]:
            if start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        self.free_regions = merged
    
    def total_free(self):
        return sum(end - start for start, end in self.free_regions)
    
    def largest_contiguous(self):
        return max((end - start) for start, end in self.free_regions) if self.free_regions else 0


class PagedAllocator:
    """Allocates in fixed-size pages (like PagedAttention)."""
    def __init__(self, total_memory_mb, page_size_mb=8):  # 8 MB ≈ 16 tokens
        self.page_size = page_size_mb
        self.total_pages = total_memory_mb // page_size_mb
        self.free_pages = self.total_pages
        self.allocations = {}  # request_id -> num_pages
    
    def allocate(self, request_id, size_mb):
        pages_needed = -(-size_mb // self.page_size)  # ceil division
        if pages_needed <= self.free_pages:
            self.free_pages -= pages_needed
            self.allocations[request_id] = pages_needed
            return True
        return False
    
    def free(self, request_id):
        if request_id in self.allocations:
            self.free_pages += self.allocations.pop(request_id)
    
    def total_free(self):
        return self.free_pages * self.page_size
        
        
TOTAL_MEMORY = 1500  # ~1.5 GB in MB (your T4's KV budget)
# Convert sequence lengths to MB using 0.5 MB/token
# seq_length × 0.5 MB ≈ KV size, but let's use shorter seqs for T4

# Simulate with proportionally scaled requests for T4
initial_requests = {
    'req1': 50, 'req2': 120, 'req3': 80, 'req4': 300,
    'req5': 40, 'req6': 250, 'req7': 60, 'req8': 180,
    'req9': 100, 'req10': 200
}

for name, AllocClass in [("Contiguous", ContiguousAllocator), ("Paged", PagedAllocator)]:
    alloc = AllocClass(TOTAL_MEMORY) if name == "Paged" else AllocClass(TOTAL_MEMORY)
    
    # Allocate initial requests
    for rid, size in initial_requests.items():
        alloc.allocate(rid, size)
    
    # Free even-numbered requests (creates holes)
    for rid in ['req2', 'req4', 'req6', 'req8']:
        alloc.free(rid)
    
    print(f"\n{name}: After freeing 4 requests:")
    print(f"  Total free: {alloc.total_free()} MB")
    if name == "Contiguous":
        print(f"  Largest contiguous: {alloc.largest_contiguous()} MB")
    
    # Try allocating new requests
    new_requests = {'new1': 250, 'new2': 350, 'new3': 400}
    for rid, size in new_requests.items():
        success = alloc.allocate(rid, size)
        print(f"  Allocate {rid} ({size} MB): {'✓' if success else '✗ FAILED'}")        