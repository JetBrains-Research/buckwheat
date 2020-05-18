var targetInterface: NIONetworkInterface? = nil
if let interfaceAddress = CommandLine.arguments.dropFirst().first,
   let targetAddress = try? SocketAddress(ipAddress: interfaceAddress, port: 0) {
    for interface in try! System.enumerateInterfaces() {
        if interface.address == targetAddress {
            targetInterface = interface
            break
        }
    }

    if targetInterface == nil {
        fatalError("Could not find interface for \(interfaceAddress)")
    }
}
