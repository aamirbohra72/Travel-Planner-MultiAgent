# TripMate Architecture

This diagram reflects the current repository implementation. Solid arrows show the request flow; dashed arrows show model calls or persistence. Flight, hotel, and place search currently use mock data. Weather uses mock data by default, with an optional Open-Meteo source.

```mermaid
flowchart LR
    traveler([Traveler])

    subgraph experience["1. User experience"]
        ui["Streamlit app<br/>Trip composer · chat history<br/>Preferences · approval controls"]
        response["Rendered plan<br/>or actionable error"]
    end

    subgraph workflow["2. LangGraph orchestration"]
        begin["Begin turn<br/>Reset per-turn state"]
        input_guard{"Input guardrail<br/>travel / off-topic / unsafe"}
        parser["Request parser<br/>Destination · dates · duration · budget"]
        supervisor["Supervisor<br/>Select next specialist"]
        flight["Flight agent"]
        hotel["Hotel agent"]
        itinerary["Itinerary agent"]
        budget["Budget agent"]
        prepare["Prepare final plan<br/>Totals · caveats · proposal"]
        approval{"Human approval interrupt<br/>Approve / edit / reject"}
        output_guard["Output guardrail<br/>Safety · budget disclosure · bookings"]
        final["Final answer"]
    end

    subgraph providers["3. Model and tool services"]
        groq[("Groq chat model<br/>qwen/qwen3.8-27b")]
        mcp_client["MultiServerMCPClient"]
        mcp_server["TripMate MCP server<br/>Streamable HTTP"]
        mock["Mock travel source<br/>Flights · hotels · places"]
        weather{"Weather source"}
        open_meteo["Open-Meteo<br/>optional weather API"]
        no_booking["No booking provider<br/>Approval does not purchase"]
    end

    subgraph persistence["4. Persistence"]
        checkpoints[("SQLite graph checkpoints<br/>Resumable threads")]
        preferences[("SQLite preference store<br/>User travel preferences")]
    end

    traveler -->|Trip request| ui
    ui -->|Request + thread context| begin
    begin --> input_guard
    input_guard -->|Travel| parser
    input_guard -->|Off-topic or unsafe| output_guard
    parser --> supervisor

    supervisor -->|Next step| flight
    supervisor -->|Next step| hotel
    supervisor -->|Next step| itinerary
    supervisor -->|Next step| budget
    flight --> supervisor
    hotel --> supervisor
    itinerary --> supervisor
    budget --> supervisor
    supervisor -->|Finish| prepare

    prepare -->|No booking requested| output_guard
    prepare -->|Booking proposal requested| approval
    approval -->|Decision recorded only| output_guard
    output_guard --> final
    final --> response
    response --> ui

    flight --> mcp_client
    hotel --> mcp_client
    itinerary --> mcp_client
    mcp_client --> mcp_server
    mcp_server --> mock
    mcp_server --> weather
    weather -->|Configured option| open_meteo

    input_guard -.-> groq
    parser -.-> groq
    supervisor -.-> groq
    flight -.-> groq
    hotel -.-> groq
    itinerary -.-> groq
    budget -.-> groq
    output_guard -.-> groq

    begin -.-> checkpoints
    supervisor -.-> checkpoints
    approval -.-> checkpoints
    ui -.->|Load / save preferences| preferences
    prepare -.-> no_booking

    classDef user fill:#dcfce7,stroke:#16a34a,color:#14532d,stroke-width:2px;
    classDef ui fill:#dbeafe,stroke:#2563eb,color:#172554,stroke-width:2px;
    classDef guard fill:#fce7f3,stroke:#db2777,color:#500724,stroke-width:2px;
    classDef flow fill:#ede9fe,stroke:#7c3aed,color:#2e1065,stroke-width:2px;
    classDef agent fill:#fef3c7,stroke:#d97706,color:#451a03,stroke-width:2px;
    classDef service fill:#cffafe,stroke:#0891b2,color:#164e63,stroke-width:2px;
    classDef mock fill:#ffedd5,stroke:#ea580c,color:#431407,stroke-width:2px;
    classDef store fill:#e0e7ff,stroke:#4f46e5,color:#1e1b4b,stroke-width:2px;
    classDef boundary fill:#fee2e2,stroke:#dc2626,color:#450a0a,stroke-width:2px;

    class traveler user;
    class ui,response ui;
    class input_guard,output_guard,approval guard;
    class begin,parser,supervisor,prepare,final flow;
    class flight,hotel,itinerary,budget agent;
    class groq,mcp_client,mcp_server,weather,open_meteo service;
    class mock mock;
    class checkpoints,preferences store;
    class no_booking boundary;
```

## Component responsibilities

| Component | Responsibility |
| --- | --- |
| Streamlit | Collects trip requests, displays conversation history, edits preferences, and presents approval controls. |
| LangGraph | Runs input checks, parses trip details, routes among specialist agents, prepares the plan, and reviews the final response. |
| Groq | Supplies the chat model used by the guardrails, parser, supervisor, and agents. |
| TripMate MCP | Exposes flight, hotel, weather, and place tools to the specialist agents. |
| SQLite | Stores resumable graph checkpoints and long-term user preferences. |
| Booking approval | Records an approve/edit/reject decision. It does not submit a reservation or payment. |

## Data and integration boundaries

- Flight, hotel, and place results are generated by `MockTravelDataSource`; they are samples, not live offers or availability.
- Weather is mock-backed by default. Setting `TRIPMATE_DATA_SOURCE=open-meteo` switches only the weather lookup to Open-Meteo.
- The configured MCP server is TripMate's own server. The Amadeus MCP server is not wired into this graph; it provides flight search only and does not provide hotel search or booking.
- A booking request creates a reviewable proposal and pauses for human input. There is no booking execution provider connected.

## Local runtime

Run the MCP server and Streamlit UI in separate terminals:

```powershell
python -m tripmate.mcp_servers.server
streamlit run streamlit_app.py
```