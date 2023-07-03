% Server_process.m
clear;
close all;

% Initialize table
global table;
table = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9; ... % Index
         0, 11, 22, 33, 44, 55, 66, 77, 88, 99]; % Data (Integers)

% Create server socket
fprintf('Creating server socket...');
tcpipServer = tcpserver('127.0.0.1', 6002);
fprintf(' CREATED\n');

% Main loop
while true
    if tcpipServer.NumBytesAvailable ~= 0
        data = read(tcpipServer, tcpipServer.NumBytesAvailable, "string");
        reply(tcpipServer, data)
    end
end

% Function to handle requests and send responses
function reply(srv, packet)
    global table;
    fields = split(packet, ';');
    op = split(fields(1), '=');
    op = op(2);
    if op ~= "CLR"
        temp = split(fields(2), '=');
        temp = temp(2);
        temp = split(temp,',');
        indices = str2double(temp) + 1; % Add 1 to indices for MATLAB's 1-based indexing
    end
    
    if op == "GET"
        data = table(2, indices);
        response = strcat("OP=GET;DATA=", join(string(data), ','), ";");
    elseif op == "PUT"
        temp = split(fields(3), '=');
        temp = split(temp(2),',');
        data = str2double(temp);
        table(2, indices) = data;
        response = "OP=PUT;SUCCESS;";
    elseif op == "CLR"
        table(2, :) = 0;
        response = "OP=CLR;SUCCESS;";
    elseif op == "ADD"
        data = sum(table(2, indices));
        response = strcat("OP=ADD;DATA=", string(data), ";");
    else
        response = "Invalid operation";
    end
    
    % Send response to proxy
    while (1)
        try
            disp("Sending response to proxy:");
            disp(response);
            srv.write(response);
            break;
        catch
            pause(0.01);
        end
    end
    
    % Print the current table state
    disp("Current table state:");
    disp(table);
end
